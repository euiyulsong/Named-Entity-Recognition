#!/usr/bin/env python3
import os, re, gc, json, time, math, random, argparse, urllib.request
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel, AutoProcessor, AutoModelForMultimodalLM
from peft import LoraConfig, get_peft_model
from torchcrf import CRF

RUNNE_TRAIN_URL = "https://raw.githubusercontent.com/dialogue-evaluation/RuNNE/refs/heads/main/public_data/train.jsonl"
RUNNE_TEST_URL = "https://raw.githubusercontent.com/dialogue-evaluation/RuNNE/refs/heads/main/public_data/test_with_answers.jsonl"

# ----------------------------- utils -----------------------------
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def device_name(): return "cuda" if torch.cuda.is_available() else "cpu"

def download(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0: return
    print("download:", url)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
        f.write(r.read())

def read_runne(path):
    rows=[]
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            x=json.loads(line)
            text=x.get("sentences", x.get("text", ""))
            ents=[]
            for e in x.get("ners", []):
                # RuNNE offsets: first char, last char (inclusive)
                s, last, lab = int(e[0]), int(e[1]), str(e[2])
                end=last+1
                if 0 <= s < end <= len(text):
                    ents.append({"start":s,"end":end,"label":lab})
            rows.append({"id":x.get("id",len(rows)),"text":text,"entities":ents})
    return rows

def subsample(rows, n, seed):
    if n is None or n <= 0 or len(rows) <= n: return rows
    idx=list(range(len(rows))); random.Random(seed).shuffle(idx)
    return [rows[i] for i in idx[:n]]

def load_runne(args):
    cache=os.path.expanduser(args.data_cache)
    train_path=os.path.join(cache,"train.jsonl")
    test_path=os.path.join(cache,"test_with_answers.jsonl")
    download(RUNNE_TRAIN_URL, train_path); download(RUNNE_TEST_URL, test_path)
    all_train=read_runne(train_path); test=read_runne(test_path)
    idx=list(range(len(all_train))); random.Random(args.seed).shuffle(idx)
    nv=max(1, round(len(idx)*args.valid_ratio)); vids=set(idx[:nv])
    train=[x for i,x in enumerate(all_train) if i not in vids]
    valid=[x for i,x in enumerate(all_train) if i in vids]
    train=subsample(train,args.max_train,args.seed+1)
    valid=subsample(valid,args.max_valid,args.seed+2)
    test=subsample(test,args.max_test,args.seed+3)
    return train,valid,test

def labels_from_train(rows):
    return sorted({e["label"] for r in rows for e in r["entities"]})

def gold_set(row, labels):
    labs=set(labels)
    return {(e["start"],e["end"],e["label"]) for e in row["entities"] if e["label"] in labs}

def prf(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else 0.; r=tp/(tp+fn) if tp+fn else 0.
    f=2*p*r/(p+r) if p+r else 0.; return p,r,f

def overlap_entity_keys(row):
    es=row["entities"]; out=set()
    for i,a in enumerate(es):
        for j,b in enumerate(es):
            if i==j: continue
            if max(a["start"],b["start"]) < min(a["end"],b["end"]):
                out.add((a["start"],a["end"],a["label"])); break
    return out

def score_docs(pred_by_doc, rows, labels):
    TP=FP=FN=0; OTP=OFP=OFN=0
    for di,row in enumerate(rows):
        pred=pred_by_doc.get(di,set()); gold=gold_set(row,labels)
        TP+=len(pred&gold); FP+=len(pred-gold); FN+=len(gold-pred)
        og=overlap_entity_keys(row) & gold
        # overlap-only: predicted entities are counted only if their span overlaps any gold nested/overlap region
        overlap_regions=[(s,e) for s,e,_ in og]
        op={x for x in pred if any(max(x[0],s)<min(x[1],e) for s,e in overlap_regions)}
        OTP+=len(op&og); OFP+=len(op-og); OFN+=len(og-op)
    p,r,f=prf(TP,FP,FN); op,orr,of=prf(OTP,OFP,OFN)
    return {"precision":p,"recall":r,"f1":f,"overlap_f1":of,"tp":TP,"fp":FP,"fn":FN}

def print_stats(train,valid,test,labels):
    print("\n"+"="*100+"\nDATASET STATISTICS\n"+"="*100)
    print("labels:",len(labels),labels)
    for name,rows in [("train",train),("valid",valid),("test",test)]:
        ne=sum(len(r["entities"]) for r in rows)
        ov=sum(bool(overlap_entity_keys(r)) for r in rows)
        print(f"{name:7s} docs={len(rows):4d} entities={ne:6d} avg={ne/max(1,len(rows)):6.2f} overlap_docs={ov:4d}")

# ----------------------- token sliding windows -----------------------
class TokenWindowDataset(Dataset):
    def __init__(self, rows, tokenizer, labels, max_length=256, stride=96):
        self.rows=rows; self.tok=tokenizer; self.labels=labels; self.l2i={x:i for i,x in enumerate(labels)}
        self.samples=[]
        for di,row in enumerate(tqdm(rows, desc="build token windows", leave=False)):
            enc=tokenizer(row["text"], truncation=True, max_length=max_length, stride=stride,
                          return_overflowing_tokens=True, return_offsets_mapping=True,
                          add_special_tokens=True)
            nw=len(enc["input_ids"])
            for wi in range(nw):
                offs=[tuple(x) for x in enc["offset_mapping"][wi]]
                real=[x for x in offs if x[1]>x[0]]
                if not real: continue
                cs=min(s for s,e in real); ce=max(e for s,e in real)
                ents=[e for e in row["entities"] if e["label"] in self.l2i and e["start"]>=cs and e["end"]<=ce]
                self.samples.append({
                    "doc_idx":di,"window_idx":wi,"num_windows":nw,
                    "input_ids":enc["input_ids"][wi],"attention_mask":enc["attention_mask"][wi],
                    "offsets":offs,"char_start":cs,"char_end":ce,"entities":ents
                })
    def __len__(self): return len(self.samples)
    def __getitem__(self,idx):
        s=self.samples[idx]; offs=s["offsets"]; T=len(offs); L=len(self.labels)
        binary=torch.zeros(T,L,dtype=torch.float32); bio=torch.zeros(T,L,dtype=torch.long)
        valid=torch.tensor([e>st for st,e in offs],dtype=torch.bool)
        for ent in s["entities"]:
            li=self.l2i[ent["label"]]; hit=[]
            for ti,(ts,te) in enumerate(offs):
                if te>ts and max(ts,ent["start"]) < min(te,ent["end"]): hit.append(ti)
            for j,ti in enumerate(hit):
                binary[ti,li]=1.; bio[ti,li]=1 if j==0 else 2
        return {**s,"input_ids":torch.tensor(s["input_ids"],dtype=torch.long),
                "attention_mask":torch.tensor(s["attention_mask"],dtype=torch.long),
                "binary":binary,"bio":bio,"valid_mask":valid}

def token_collate(batch,pad_id):
    B=len(batch); M=max(len(x["input_ids"]) for x in batch); L=batch[0]["binary"].shape[-1]
    ids=torch.full((B,M),pad_id,dtype=torch.long); am=torch.zeros(B,M,dtype=torch.long)
    binary=torch.zeros(B,M,L); bio=torch.zeros(B,M,L,dtype=torch.long); valid=torch.zeros(B,M,dtype=torch.bool)
    out={"doc_idx":[],"window_idx":[],"num_windows":[],"offsets":[],"char_start":[],"char_end":[]}
    for i,x in enumerate(batch):
        n=len(x["input_ids"]); ids[i,:n]=x["input_ids"]; am[i,:n]=x["attention_mask"]
        binary[i,:n]=x["binary"]; bio[i,:n]=x["bio"]; valid[i,:n]=x["valid_mask"]
        for k in out: out[k].append(x[k])
    return {"input_ids":ids,"attention_mask":am,"binary":binary,"bio":bio,"valid_mask":valid,**out}

# ----------------------------- models -----------------------------
class EncoderSigmoid(nn.Module):
    def __init__(self,name,nlabels):
        super().__init__(); self.enc=AutoModel.from_pretrained(name); h=self.enc.config.hidden_size
        self.drop=nn.Dropout(.1); self.head=nn.Linear(h,nlabels)
    def forward(self,ids,mask): return self.head(self.drop(self.enc(input_ids=ids,attention_mask=mask).last_hidden_state))

class EncoderMultiCRF(nn.Module):
    def __init__(self,name,nlabels,aux_weight=.5):
        super().__init__(); self.enc=AutoModel.from_pretrained(name); h=self.enc.config.hidden_size
        self.nlabels=nlabels; self.head=nn.Linear(h,nlabels*3); self.crfs=nn.ModuleList([CRF(3,batch_first=True) for _ in range(nlabels)])
        self.aux_weight=aux_weight
    def emissions(self,ids,mask):
        h=self.enc(input_ids=ids,attention_mask=mask).last_hidden_state; B,T,_=h.shape
        return self.head(h).view(B,T,self.nlabels,3)
    def loss(self,ids,mask,tags,valid_mask):
        em=self.emissions(ids,mask); cmask=mask.bool(); losses=[]
        # CRF NLL + weighted auxiliary CE to fight O-class collapse.
        for li in range(self.nlabels):
            nll=-self.crfs[li](em[:,:,li,:],tags[:,:,li],mask=cmask,reduction="mean")
            v=valid_mask
            y=tags[:,:,li][v]; z=em[:,:,li,:][v]
            if y.numel():
                cnt=torch.bincount(y,minlength=3).float(); w=(cnt.sum()/(cnt+1.)).clamp(1.,20.); w=w/w[0].clamp_min(1e-6)
                ce=F.cross_entropy(z,y,weight=w)
            else: ce=torch.tensor(0.,device=ids.device)
            losses.append(nll+self.aux_weight*ce)
        return torch.stack(losses).mean()
    @torch.no_grad()
    def decode(self,ids,mask):
        em=self.emissions(ids,mask); B,T,L,_=em.shape; out=torch.zeros(B,T,L,dtype=torch.long,device=ids.device); m=mask.bool()
        for li in range(L):
            paths=self.crfs[li].decode(em[:,:,li,:],mask=m)
            for b,p in enumerate(paths): out[b,:len(p),li]=torch.tensor(p,device=ids.device)
        return out

# ------------------------ token span decoding ------------------------
def sigmoid_spans(prob,offs,labels,thr):
    pred=prob>=thr; spans=set(); T,L=pred.shape
    for li,lab in enumerate(labels):
        st=en=None
        for t in range(min(T,len(offs))):
            s,e=offs[t]; pos=(e>s and bool(pred[t,li]))
            if pos:
                if st is None: st=s
                en=e
            elif st is not None:
                spans.add((int(st),int(en),lab)); st=en=None
        if st is not None: spans.add((int(st),int(en),lab))
    return spans

def bio_spans(tags,offs,labels):
    spans=set(); T,L=tags.shape
    for li,lab in enumerate(labels):
        st=en=None
        for t in range(min(T,len(offs))):
            s,e=offs[t]
            if e<=s: continue
            tag=int(tags[t,li])
            if tag==1:
                if st is not None: spans.add((int(st),int(en),lab))
                st,en=s,e
            elif tag==2:
                if st is None: st=s
                en=e
            else:
                if st is not None: spans.add((int(st),int(en),lab)); st=en=None
        if st is not None: spans.add((int(st),int(en),lab))
    return spans

def accept_window_span(span, wi, nw, cs, ce):
    # discard chunk-edge fragments; overlapping adjacent windows recover them.
    s,e,_=span
    if wi>0 and s<=cs: return False
    if wi<nw-1 and e>=ce: return False
    return True

@torch.no_grad()
def eval_sigmoid(model,loader,rows,labels,device,thr):
    model.eval(); pred=defaultdict(set); t0=time.perf_counter()
    for b in loader:
        ids=b["input_ids"].to(device); am=b["attention_mask"].to(device)
        probs=torch.sigmoid(model(ids,am)).cpu().numpy()
        for i in range(len(b["doc_idx"])):
            ss=sigmoid_spans(probs[i],b["offsets"][i],labels,thr)
            ss={x for x in ss if accept_window_span(x,b["window_idx"][i],b["num_windows"][i],b["char_start"][i],b["char_end"][i])}
            pred[b["doc_idx"][i]].update(ss)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    out=score_docs(pred,rows,labels); out["latency"]=(time.perf_counter()-t0)/max(1,len(rows)); return out

@torch.no_grad()
def eval_crf(model,loader,rows,labels,device):
    model.eval(); pred=defaultdict(set); t0=time.perf_counter()
    for b in loader:
        ids=b["input_ids"].to(device); am=b["attention_mask"].to(device); tags=model.decode(ids,am).cpu().numpy()
        for i in range(len(b["doc_idx"])):
            ss=bio_spans(tags[i],b["offsets"][i],labels)
            ss={x for x in ss if accept_window_span(x,b["window_idx"][i],b["num_windows"][i],b["char_start"][i],b["char_end"][i])}
            pred[b["doc_idx"][i]].update(ss)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    out=score_docs(pred,rows,labels); out["latency"]=(time.perf_counter()-t0)/max(1,len(rows)); return out

def compute_pos_weight(loader,nlabels,max_w=50.):
    pos=torch.zeros(nlabels); total=0
    for b in loader:
        v=b["valid_mask"]; y=b["binary"]
        pos += (y*v.unsqueeze(-1)).sum((0,1)); total += int(v.sum())
    neg=torch.tensor(float(total))-pos
    return (neg/(pos+1.)).clamp(1.,max_w)

def train_sigmoid(model,train_loader,valid_loader,valid_rows,labels,device,args):
    model.to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.bert_lr,weight_decay=.01)
    pw=compute_pos_weight(train_loader,len(labels),args.max_pos_weight).to(device)
    print("pos_weight min/mean/max:",float(pw.min()),float(pw.mean()),float(pw.max()))
    best=(-1,None,.5); thresholds=[float(x) for x in args.thresholds.split(",")]
    for ep in range(args.bert_epochs):
        model.train(); bar=tqdm(train_loader,desc=f"SIGMOID {ep+1}/{args.bert_epochs}")
        for b in bar:
            ids=b["input_ids"].to(device); am=b["attention_mask"].to(device); y=b["binary"].to(device); v=b["valid_mask"].to(device)
            z=model(ids,am); loss=F.binary_cross_entropy_with_logits(z,y,pos_weight=pw,reduction="none"); loss=loss[v.unsqueeze(-1).expand_as(loss)].mean()
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); bar.set_postfix(loss=f"{loss.item():.4f}")
        for th in thresholds:
            r=eval_sigmoid(model,valid_loader,valid_rows,labels,device,th)
            print(f"[SIGMOID] ep={ep+1} th={th:.2f} P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f} overlapF1={r['overlap_f1']:.4f}")
            if r["f1"]>best[0]: best=(r["f1"],{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},th)
    model.load_state_dict(best[1]); model.to(device); print("best sigmoid val F1/thr:",best[0],best[2]); return model,best[2]

def train_crf(model,train_loader,valid_loader,valid_rows,labels,device,args):
    model.to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.bert_lr,weight_decay=.01); best=(-1,None)
    for ep in range(args.bert_epochs):
        model.train(); bar=tqdm(train_loader,desc=f"MULTI-CRF {ep+1}/{args.bert_epochs}")
        for b in bar:
            ids=b["input_ids"].to(device); am=b["attention_mask"].to(device); y=b["bio"].to(device); v=b["valid_mask"].to(device)
            loss=model.loss(ids,am,y,v); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); bar.set_postfix(loss=f"{loss.item():.3f}")
        r=eval_crf(model,valid_loader,valid_rows,labels,device)
        print(f"[CRF] ep={ep+1} P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f} overlapF1={r['overlap_f1']:.4f}")
        if r["f1"]>best[0]: best=(r["f1"],{k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    model.load_state_dict(best[1]); model.to(device); print("best CRF val F1:",best[0]); return model

# ----------------------------- Qwen chunks -----------------------------
def char_chunks(row, chunk_size=900, overlap=150):
    text=row["text"]; out=[]; start=0; ci=0
    while start < len(text):
        end=min(len(text),start+chunk_size)
        # Prefer whitespace boundary near right edge.
        if end < len(text):
            lo=max(start+chunk_size//2,end-120); cut=max(text.rfind("\n",lo,end),text.rfind(" ",lo,end))
            if cut>lo: end=cut
        ents=[]
        for e in row["entities"]:
            if e["start"]>=start and e["end"]<=end:
                ents.append({"start":e["start"]-start,"end":e["end"]-start,"label":e["label"]})
        out.append({"doc_start":start,"doc_end":end,"chunk_idx":ci,"text":text[start:end],"entities":ents})
        ci+=1
        if end>=len(text): break
        nxt=max(start+1,end-overlap); start=nxt
    n=len(out)
    for x in out: x["num_chunks"]=n
    return out

def qwen_prompt(text,labels):
    return ("Extract all named entities from TEXT. Nested and overlapping entities are allowed.\n"
            "Allowed labels: "+", ".join(labels)+"\n"
            "Offsets are relative to TEXT below; start inclusive, end exclusive.\n"
            "Return ONLY JSON: [{\"start\":0,\"end\":4,\"label\":\"PERSON\"}]. If none, return [].\nTEXT:\n"+text)

def target_json(ents,labels):
    labs=set(labels); x=[{"start":int(e["start"]),"end":int(e["end"]),"label":e["label"]} for e in ents if e["label"] in labs]
    x.sort(key=lambda z:(z["start"],z["end"],z["label"])); return json.dumps(x,ensure_ascii=False,separators=(",",":"))

def parse_json_entities(raw,labels,text):
    raw=raw.strip(); raw=re.sub(r"^```(?:json)?\s*","",raw,flags=re.I); raw=re.sub(r"\s*```$","",raw)
    m=re.search(r"\[.*\]",raw,re.S)
    if m: raw=m.group(0)
    try: data=json.loads(raw)
    except Exception: return set(),False
    if not isinstance(data,list): return set(),False
    labs=set(labels); out=set()
    for x in data:
        if not isinstance(x,dict): continue
        try: s=int(x["start"]); e=int(x["end"]); lab=str(x["label"])
        except Exception: continue
        if lab in labs and 0<=s<e<=len(text): out.add((s,e,lab))
    return out,True

def load_qwen(name):
    proc=AutoProcessor.from_pretrained(name)
    if proc.tokenizer.pad_token_id is None: proc.tokenizer.pad_token=proc.tokenizer.eos_token
    dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32)
    model=AutoModelForMultimodalLM.from_pretrained(name,dtype=dtype,device_map="auto" if torch.cuda.is_available() else None)
    return model,proc

def render_user(proc,prompt,add_generation=True):
    msgs=[{"role":"user","content":[{"type":"text","text":prompt}]}]
    return proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=add_generation)

def qwen_generate(model,proc,text,labels,max_new_tokens):
    msgs=[{"role":"user","content":[{"type":"text","text":qwen_prompt(text,labels)}]}]
    inp=proc.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors="pt")
    dev=next(model.parameters()).device; inp={k:v.to(dev) for k,v in inp.items() if torch.is_tensor(v)}; n=inp["input_ids"].shape[1]
    out=model.generate(**inp,max_new_tokens=max_new_tokens,do_sample=False,pad_token_id=proc.tokenizer.pad_token_id)
    return proc.decode(out[0,n:],skip_special_tokens=True,clean_up_tokenization_spaces=False)

@torch.no_grad()
def eval_qwen(model,proc,rows,labels,args,desc):
    model.eval(); pred=defaultdict(set); parse_fail=0; nch=0; t0=time.perf_counter(); examples=[]
    for di,row in enumerate(tqdm(rows,desc=desc)):
        chunks=char_chunks(row,args.qwen_chunk_chars,args.qwen_chunk_overlap)
        for ch in chunks:
            raw=qwen_generate(model,proc,ch["text"],labels,args.qwen_max_new_tokens); local,ok=parse_json_entities(raw,labels,ch["text"])
            if not ok: parse_fail+=1
            nch+=1
            for s,e,lab in local:
                gs,ge=s+ch["doc_start"],e+ch["doc_start"]
                # edge fragment filtering
                if ch["chunk_idx"]>0 and s<=0: continue
                if ch["chunk_idx"]<ch["num_chunks"]-1 and e>=len(ch["text"]): continue
                pred[di].add((gs,ge,lab))
            if len(examples)<5: examples.append({"doc":di,"chunk":ch["chunk_idx"],"gold":ch["entities"],"raw":raw[:800]})
    if torch.cuda.is_available(): torch.cuda.synchronize()
    r=score_docs(pred,rows,labels); r["latency"]=(time.perf_counter()-t0)/max(1,len(rows)); r["parse_failure_rate"]=parse_fail/max(1,nch); r["chunks"]=nch; r["examples"]=examples; return r

class QwenChunkDataset(Dataset):
    def __init__(self,rows,proc,labels,args):
        self.proc=proc; self.tok=proc.tokenizer; self.labels=labels; self.maxlen=args.qwen_max_length; self.samples=[]
        for r in rows: self.samples.extend(char_chunks(r,args.qwen_chunk_chars,args.qwen_chunk_overlap))
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        ch=self.samples[i]; prompt=qwen_prompt(ch["text"],self.labels); target=target_json(ch["entities"],self.labels)
        user=[{"role":"user","content":[{"type":"text","text":prompt}]}]
        full=user+[{"role":"assistant","content":[{"type":"text","text":target}]}]
        ptxt=self.proc.apply_chat_template(user,tokenize=False,add_generation_prompt=True)
        ftxt=self.proc.apply_chat_template(full,tokenize=False,add_generation_prompt=False)
        pids=self.tok(ptxt,add_special_tokens=False,truncation=True,max_length=self.maxlen)["input_ids"]
        fids=self.tok(ftxt,add_special_tokens=False,truncation=True,max_length=self.maxlen)["input_ids"]
        ids=torch.tensor(fids,dtype=torch.long); labs=ids.clone(); labs[:min(len(pids),len(fids))]=-100
        return {"input_ids":ids,"labels":labs}

class CausalCollator:
    def __init__(self,pad): self.pad=pad
    def __call__(self,b):
        M=max(len(x["input_ids"]) for x in b); B=len(b)
        ids=torch.full((B,M),self.pad,dtype=torch.long); am=torch.zeros(B,M,dtype=torch.long); labs=torch.full((B,M),-100,dtype=torch.long)
        for i,x in enumerate(b):
            n=len(x["input_ids"]); ids[i,:n]=x["input_ids"]; am[i,:n]=1; labs[i,:n]=x["labels"]
        return {"input_ids":ids,"attention_mask":am,"labels":labs}

def add_lora(model,r):
    cand=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","in_proj_qkv","in_proj_z","in_proj_a","in_proj_b","out_proj"]
    existing={n.split(".")[-1] for n,m in model.named_modules() if isinstance(m,nn.Linear)}
    targets=[x for x in cand if x in existing]
    if not targets: raise RuntimeError("Could not find LoRA target Linear modules")
    print("LoRA targets:",targets)
    cfg=LoraConfig(r=r,lora_alpha=2*r,lora_dropout=.05,bias="none",task_type="CAUSAL_LM",target_modules=targets)
    model=get_peft_model(model,cfg); model.print_trainable_parameters(); return model

def train_qwen(model,proc,rows,labels,args):
    model=add_lora(model,args.lora_r)
    if hasattr(model.config,"use_cache"): model.config.use_cache=False
    try: model.gradient_checkpointing_enable()
    except Exception: pass
    ds=QwenChunkDataset(rows,proc,labels,args); dl=DataLoader(ds,batch_size=args.qwen_batch_size,shuffle=True,collate_fn=CausalCollator(proc.tokenizer.pad_token_id),num_workers=0)
    params=[p for p in model.parameters() if p.requires_grad]; opt=torch.optim.AdamW(params,lr=args.qwen_lr,weight_decay=.01)
    use_amp=torch.cuda.is_available(); accum=max(1,args.qwen_grad_acc); opt.zero_grad(set_to_none=True)
    for ep in range(args.qwen_epochs):
        model.train(); bar=tqdm(dl,desc=f"QWEN-LORA {ep+1}/{args.qwen_epochs}")
        for step,b in enumerate(bar,1):
            dev=next(model.parameters()).device; b={k:v.to(dev) for k,v in b.items()}
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=use_amp and torch.cuda.is_bf16_supported()):
                out=model(**b); loss=out.loss/accum
            loss.backward()
            if step%accum==0 or step==len(dl):
                torch.nn.utils.clip_grad_norm_(params,1.); opt.step(); opt.zero_grad(set_to_none=True)
            bar.set_postfix(loss=f"{loss.item()*accum:.4f}")
    if hasattr(model.config,"use_cache"): model.config.use_cache=True
    return model

# ----------------------------- result -----------------------------
def print_results(results):
    print("\n"+"="*125+"\nFINAL RESULT\n"+"="*125)
    print(f"{'model':32s}{'P':>9s}{'R':>9s}{'F1':>9s}{'overlapF1':>12s}{'ms/doc':>12s}{'json_fail':>12s}")
    print("-"*125)
    for name,r in results.items():
        jf=r.get("parse_failure_rate",None); j="-" if jf is None else f"{jf:.4f}"
        print(f"{name:32s}{r['precision']:9.4f}{r['recall']:9.4f}{r['f1']:9.4f}{r.get('overlap_f1',0):12.4f}{r.get('latency',0)*1000:12.2f}{j:>12s}")
    print("="*125)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-cache",default="~/.cache/runne_nested_ner_v2")
    ap.add_argument("--max-train",type=int,default=None); ap.add_argument("--max-valid",type=int,default=None); ap.add_argument("--max-test",type=int,default=None)
    ap.add_argument("--valid-ratio",type=float,default=.10); ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--bert-model",default="FacebookAI/xlm-roberta-base"); ap.add_argument("--bert-epochs",type=int,default=5); ap.add_argument("--bert-batch-size",type=int,default=16); ap.add_argument("--bert-lr",type=float,default=3e-5)
    ap.add_argument("--bert-max-length",type=int,default=256); ap.add_argument("--bert-stride",type=int,default=96); ap.add_argument("--thresholds",default="0.15,0.20,0.25,0.30,0.35,0.40,0.50"); ap.add_argument("--max-pos-weight",type=float,default=50.)
    ap.add_argument("--crf-aux-weight",type=float,default=.5)
    ap.add_argument("--qwen-model",default="Qwen/Qwen3.5-0.8B"); ap.add_argument("--qwen-epochs",type=int,default=3); ap.add_argument("--qwen-batch-size",type=int,default=2); ap.add_argument("--qwen-grad-acc",type=int,default=8); ap.add_argument("--qwen-lr",type=float,default=1e-4); ap.add_argument("--lora-r",type=int,default=16)
    ap.add_argument("--qwen-chunk-chars",type=int,default=900); ap.add_argument("--qwen-chunk-overlap",type=int,default=150); ap.add_argument("--qwen-max-length",type=int,default=1536); ap.add_argument("--qwen-max-new-tokens",type=int,default=256)
    ap.add_argument("--skip-sigmoid",action="store_true"); ap.add_argument("--skip-crf",action="store_true"); ap.add_argument("--skip-zero-shot",action="store_true"); ap.add_argument("--skip-qwen-ft",action="store_true")
    args=ap.parse_args(); seed_all(args.seed); dev=device_name()
    print("="*100+"\nMULTI-LABEL / NESTED NER BENCHMARK\n"+"="*100); print("device:",dev)
    train,valid,test=load_runne(args); labels=labels_from_train(train); print_stats(train,valid,test,labels); results={}

    if not args.skip_sigmoid or not args.skip_crf:
        tok=AutoTokenizer.from_pretrained(args.bert_model,use_fast=True)
        if tok.pad_token_id is None: tok.pad_token=tok.eos_token or tok.sep_token
        trds=TokenWindowDataset(train,tok,labels,args.bert_max_length,args.bert_stride)
        vds=TokenWindowDataset(valid,tok,labels,args.bert_max_length,args.bert_stride)
        tds=TokenWindowDataset(test,tok,labels,args.bert_max_length,args.bert_stride)
        coll=lambda b: token_collate(b,tok.pad_token_id)
        tr=DataLoader(trds,batch_size=args.bert_batch_size,shuffle=True,collate_fn=coll,num_workers=0)
        va=DataLoader(vds,batch_size=args.bert_batch_size,shuffle=False,collate_fn=coll,num_workers=0)
        te=DataLoader(tds,batch_size=args.bert_batch_size,shuffle=False,collate_fn=coll,num_workers=0)
        print(f"token windows train/valid/test = {len(trds)}/{len(vds)}/{len(tds)}")

    if not args.skip_sigmoid:
        print("\n"+"="*100+"\n1. XLM-R + MULTILABEL SIGMOID\n"+"="*100)
        m=EncoderSigmoid(args.bert_model,len(labels)); m,thr=train_sigmoid(m,tr,va,valid,labels,dev,args); results["XLM-R + sigmoid"]=eval_sigmoid(m,te,test,labels,dev,thr)
        del m; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if not args.skip_crf:
        print("\n"+"="*100+"\n2. XLM-R + LABEL-WISE BIO CRF\n"+"="*100)
        m=EncoderMultiCRF(args.bert_model,len(labels),args.crf_aux_weight); m=train_crf(m,tr,va,valid,labels,dev,args); results["XLM-R + multi-CRF"]=eval_crf(m,te,test,labels,dev)
        del m; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    try: del tr,va,te,trds,vds,tds,tok
    except Exception: pass
    gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if not args.skip_zero_shot or not args.skip_qwen_ft:
        print("\n"+"="*100+"\nLOADING QWEN3.5-0.8B\n"+"="*100)
        qm,qp=load_qwen(args.qwen_model)

    if not args.skip_zero_shot:
        print("\n"+"="*100+"\n3. QWEN3.5-0.8B ZERO-SHOT (CHUNKED)\n"+"="*100)
        r=eval_qwen(qm,qp,test,labels,args,"Qwen zero-shot"); results["Qwen3.5-0.8B zero-shot"]=r
        print("zero-shot examples:"); [print(x) for x in r["examples"][:3]]

    if not args.skip_qwen_ft:
        print("\n"+"="*100+"\n4. QWEN3.5-0.8B LoRA (CHUNKED)\n"+"="*100)
        qm=train_qwen(qm,qp,train,labels,args); r=eval_qwen(qm,qp,test,labels,args,"Qwen LoRA eval"); results["Qwen3.5-0.8B LoRA"]=r
        print("LoRA examples:"); [print(x) for x in r["examples"][:3]]

    print_results(results)

if __name__=="__main__": main()
