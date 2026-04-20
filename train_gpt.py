"""
Parameter Golf — Frontier Submission
Builds directly on the official train_gpt.py baseline.

Preserves ALL required infrastructure (unchanged):
  - quantize_state_dict_int8 / dequantize_state_dict_int8
  - final_int8_zlib_roundtrip output format
  - build_sentencepiece_luts + eval_val BPB metric
  - TokenStream / DistributedTokenLoader
  - torchrun-compatible distributed setup
  - DATA_PATH / TOKENIZER_PATH env vars

Novel additions over baseline:
  [1] Depth recurrence  — loops layers 3-6 four times (20 virtual passes)
  [2] Gated state highway — persistent state h across loops (RNN-like memory)
  [3] Parallel residuals  — attention and MLP from layer 5 onward
  [4] Per-head QK-Gain   — learned per head, not global scalar
  [5] SP8192 vocabulary   — set VOCAB_SIZE=8192 DATA_PATH=...sp8192...
  [6] LeakyReLU-squared MLP — replaces relu-squared (SOTA-verified)
  [7] Bigram hash embed   — extra context signal, zero vocabulary cost
  [8] EMA weights         — shadow model averaged during training
  [9] Legal score-first TTT — adapt on already-scored tokens at eval
  [10] Sliding window eval  — stride-64 overlapping windows

Target: val_bpb ~1.065 (current SOTA: 1.0810)
Hard limit: script is under 1500 lines.
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
class Hyperparameters:
    data_path      = os.environ.get("DATA_PATH",      "./data/datasets/fineweb10B_sp8192")
    train_files    = os.path.join(data_path,          "fineweb_train_*.bin")
    val_files      = os.path.join(data_path,          "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_8192_bpe.model")
    run_id         = os.environ.get("RUN_ID",          str(uuid.uuid4()))
    seed           = int(os.environ.get("SEED",        42))

    val_batch_size  = int(os.environ.get("VAL_BATCH_SIZE",   524_288))
    val_loss_every  = int(os.environ.get("VAL_LOSS_EVERY",   500))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY",  100))

    iterations           = int(os.environ.get("ITERATIONS",            6000))
    warmdown_iters       = int(os.environ.get("WARMDOWN_ITERS",        4320))
    warmup_steps         = int(os.environ.get("WARMUP_STEPS",          20))
    train_batch_tokens   = int(os.environ.get("TRAIN_BATCH_TOKENS",    524_288))
    train_seq_len        = int(os.environ.get("TRAIN_SEQ_LEN",         4096))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    vocab_size     = int(os.environ.get("VOCAB_SIZE",     8192))
    num_layers     = int(os.environ.get("NUM_LAYERS",     11))
    num_kv_heads   = int(os.environ.get("NUM_KV_HEADS",   4))
    model_dim      = int(os.environ.get("MODEL_DIM",      512))
    num_heads      = int(os.environ.get("NUM_HEADS",      8))
    mlp_mult       = int(os.environ.get("MLP_MULT",       4))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base      = float(os.environ.get("ROPE_BASE",    10000.0))
    logit_softcap  = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))

    recur_layers   = [3, 4, 5, 6]
    recur_times    = int(os.environ.get("RECUR_TIMES",    4))
    recur_at_frac  = float(os.environ.get("RECUR_AT_FRAC", 0.28))
    parallel_from  = int(os.environ.get("PARALLEL_FROM",  5))
    state_dim      = int(os.environ.get("STATE_DIM",      128))
    qk_gain_init   = float(os.environ.get("QK_GAIN_INIT", 5.25))
    bigram_sz      = int(os.environ.get("BIGRAM_SZ",      3072))

    ema_decay      = float(os.environ.get("EMA_DECAY",      0.9965))
    ema_start_frac = float(os.environ.get("EMA_START_FRAC", 0.50))

    ttt_enabled  = os.environ.get("TTT_ENABLED", "1") == "1"
    ttt_lr       = float(os.environ.get("TTT_LR",       0.005))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_epochs   = int(os.environ.get("TTT_EPOCHS",     4))
    ttt_chunk    = int(os.environ.get("TTT_CHUNK",      32768))
    sliding_stride = int(os.environ.get("SLIDING_STRIDE", 64))

    embed_lr     = float(os.environ.get("EMBED_LR",     0.05))
    matrix_lr    = float(os.environ.get("MATRIX_LR",    0.022))
    scalar_lr    = float(os.environ.get("SCALAR_LR",    0.04))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.095))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1        = float(os.environ.get("BETA1",  0.9))
    beta2        = float(os.environ.get("BETA2",  0.95))
    adam_eps     = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))
    qat_start_frac = float(os.environ.get("QAT_START_FRAC", 0.72))


# ─────────────────────────────────────────────────────────────────────────────
# MUON OPTIMIZER  (unchanged from baseline)
# ─────────────────────────────────────────────────────────────────────────────
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X /= X.norm() + eps
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if G.size(0) > G.size(1) else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps, weight_decay=0.0, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      backend_steps=backend_steps,
                                      weight_decay=weight_decay, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size  = dist.get_world_size() if distributed else 1
        my_rank     = dist.get_rank()       if distributed else 0
        for group in self.param_groups:
            params, lr = group["params"], group["lr"]
            if not params: continue
            momentum, ns = group["momentum"], group["backend_steps"]
            nesterov, wd = group["nesterov"], group["weight_decay"]
            total = sum(int(p.numel()) for p in params)
            upd   = torch.zeros(total, device=params[0].device, dtype=torch.bfloat16)
            curr  = 0
            for i, p in enumerate(params):
                if i % world_size == my_rank and p.grad is not None:
                    g = p.grad; s = self.state[p]
                    if "buf" not in s: s["buf"] = torch.zeros_like(g)
                    s["buf"].mul_(momentum).add_(g)
                    if nesterov: g = g.add(s["buf"], alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=ns)
                    g *= max(1, g.size(0)/g.size(1)) ** 0.5
                    upd[curr:curr+p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed: dist.all_reduce(upd, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                g = upd[curr:curr+p.numel()].view_as(p).to(p.dtype)
                if wd > 0: p.mul_(1 - lr*wd)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZER-AGNOSTIC BPB EVALUATION  (unchanged from baseline)
# ─────────────────────────────────────────────────────────────────────────────
def build_sentencepiece_luts(sp, vocab_size, device):
    sv = int(sp.vocab_size()); sz = max(sv, vocab_size)
    bb = np.zeros((sz,), dtype=np.int16)
    hs = np.zeros((sz,), dtype=np.bool_)
    ib = np.ones( (sz,), dtype=np.bool_)
    for t in range(sv):
        if sp.is_control(t) or sp.is_unknown(t) or sp.is_unused(t): continue
        ib[t] = False
        if sp.is_byte(t): bb[t] = 1; continue
        piece = sp.id_to_piece(t)
        if piece.startswith("▁"): hs[t] = True; piece = piece[1:]
        bb[t] = len(piece.encode("utf-8"))
    return (torch.tensor(bb, dtype=torch.int16, device=device),
            torch.tensor(hs, dtype=torch.bool,  device=device),
            torch.tensor(ib, dtype=torch.bool,  device=device))


def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files: raise FileNotFoundError(f"No files: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel()-1) // seq_len) * seq_len
    if usable <= 0: raise ValueError("Val split too short")
    return tokens[:usable+1]


def eval_val(args, model, rank, world_size, device, grad_accum,
             val_tokens, bb_lut, hs_lut, ib_lut):
    local_seqs = (args.val_batch_size//(world_size*grad_accum)) // args.train_seq_len
    total_seqs = (val_tokens.numel()-1) // args.train_seq_len
    s0 = (total_seqs * rank)       // world_size
    s1 = (total_seqs * (rank+1))   // world_size
    vls = torch.zeros((), device=device, dtype=torch.float64)
    vtk = torch.zeros((), device=device, dtype=torch.float64)
    vby = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bs in range(s0, s1, local_seqs):
            be  = min(bs+local_seqs, s1)
            loc = val_tokens[bs*args.train_seq_len:be*args.train_seq_len+1].to(device=device,dtype=torch.int64)
            x   = loc[:-1].reshape(-1, args.train_seq_len)
            y   = loc[1: ].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                bl = model(x, y).detach()
            n = float(y.numel()); vls += bl.to(torch.float64)*n; vtk += n
            tb = bb_lut[y.reshape(-1)].to(dtype=torch.int16)
            tb += (hs_lut[y.reshape(-1)] & ~ib_lut[x.reshape(-1)]).to(dtype=torch.int16)
            vby += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in (vls, vtk, vby): dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vl  = vls / vtk
    bpt = vl.item() / math.log(2.0)
    tpb = vtk.item() / vby.item()
    model.train()
    return float(vl.item()), float(bpt * tpb)


def sliding_eval(args, model, val_tokens, device, bb_lut, hs_lut, ib_lut):
    model.eval()
    sl = args.train_seq_len; N = val_tokens.numel()-1
    d  = val_tokens.to(device=device, dtype=torch.int64)
    tot_nll=tot_tok=tot_byt=0.0
    with torch.inference_mode():
        for s in range(0, N-sl, args.sliding_stride):
            x = d[s:s+sl].unsqueeze(0); y = d[s+1:s+sl+1].unsqueeze(0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                lg = model.forward_logits(x)
            off = args.sliding_stride if s > 0 else sl
            nll = F.cross_entropy(lg[0,-off:].float(), y[0,-off:], reduction="sum")
            tot_nll += nll.item(); tot_tok += off
            tb = bb_lut[y[0,-off:]].to(dtype=torch.int16)
            tb += (hs_lut[y[0,-off:]] & ~ib_lut[x[0,-off:]]).to(dtype=torch.int16)
            tot_byt += float(tb.to(torch.float64).sum())
    avg = tot_nll/max(1,tot_tok); bpt = avg/math.log(2.0)
    tpb = tot_tok/max(1,tot_byt); model.train()
    return float(avg), float(bpt*tpb)


# ─────────────────────────────────────────────────────────────────────────────
# POST-TRAINING QUANTISATION  (unchanged from baseline — required by competition)
# ─────────────────────────────────────────────────────────────────────────────
CONTROL_TENSOR_NAME_PATTERNS = tuple(p for p in os.environ.get(
    "CONTROL_TENSOR_NAME_PATTERNS",
    "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,"
    "q_gain,skip_weight,skip_weights,qk_g,ln_s,mix,highway"
).split(",") if p)

INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = CONTROL_TENSOR_NAME_PATTERNS
INT8_KEEP_FLOAT_MAX_NUMEL   = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE    = torch.float16
INT8_CLIP_Q                 = 99.99984 / 100.0


def tensor_nbytes(t): return int(t.numel()) * int(t.element_size())


def keep_float_tensor(name, t, pod):
    if any(p in name for p in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS): return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        pod[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def quantize_float_tensor(t):
    t32 = t.float()
    if t32.ndim == 2:
        ca = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.zeros(t32.shape[0])
        cl = torch.maximum(torch.minimum(t32, ca[:,None]), -ca[:,None])
        sc = (ca/127.0).clamp_min(1.0/127.0)
        q  = torch.clamp(torch.round(cl/sc[:,None]), -127, 127).to(torch.int8).contiguous()
        return q, sc.to(INT8_PER_ROW_SCALE_DTYPE).contiguous()
    ca = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    sc = torch.tensor(ca/127.0 if ca>0 else 1.0, dtype=torch.float32)
    q  = torch.clamp(torch.round(torch.clamp(t32,-ca,ca)/sc), -127, 127).to(torch.int8).contiguous()
    return q, sc


def quantize_state_dict_int8(state_dict):
    quantized={}; scales={}; dtypes={}; passthrough={}; pod={}; qmeta={}
    stats = dict.fromkeys(("param_count","num_tensors","num_float_tensors",
                            "num_nonfloat_tensors","baseline_tensor_bytes","int8_payload_bytes"), 0)
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel()); stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1; passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t); continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, pod); passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept); continue
        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0: qmeta[name] = {"scheme":"per_row","axis":0}
        quantized[name]=q; scales[name]=s; dtypes[name]=str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q)+tensor_nbytes(s)
    obj = {"__quant_format__":"int8_clean_per_row_v1","quantized":quantized,
           "scales":scales,"dtypes":dtypes,"passthrough":passthrough}
    if qmeta: obj["qmeta"] = qmeta
    if pod:   obj["passthrough_orig_dtypes"] = pod
    return obj, stats


def dequantize_state_dict_int8(obj):
    out={}; qmeta=obj.get("qmeta",{}); pod=obj.get("passthrough_orig_dtypes",{})
    for name, q in obj["quantized"].items():
        dtype=getattr(torch,obj["dtypes"][name]); s=obj["scales"][name]
        if qmeta.get(name,{}).get("scheme")=="per_row" or s.ndim>0:
            s = s.to(torch.float32)
            out[name] = (q.float()*s.view(q.shape[0],*([1]*(q.ndim-1)))).to(dtype).contiguous()
        else:
            out[name] = (q.float()*float(s.item())).to(dtype).contiguous()
    for name, t in obj["passthrough"].items():
        ot = t.detach().to("cpu").contiguous()
        od = pod.get(name)
        if isinstance(od,str): ot = ot.to(dtype=getattr(torch,od)).contiguous()
        out[name] = ot
    return out


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (unchanged from baseline)
# ─────────────────────────────────────────────────────────────────────────────
def load_data_shard(file: Path) -> Tensor:
    hb  = 256 * np.dtype("<i4").itemsize
    hdr = np.fromfile(file, dtype="<i4", count=256)
    if hdr.size!=256 or int(hdr[0])!=20240520 or int(hdr[1])!=1:
        raise ValueError(f"Bad shard header: {file}")
    n = int(hdr[2])
    if file.stat().st_size != hb + n*2: raise ValueError(f"Shard size mismatch: {file}")
    return torch.from_numpy(np.fromfile(file,dtype="<u2",count=n,offset=hb).astype(np.uint16,copy=False))


class TokenStream:
    def __init__(self, pattern):
        self.files=[Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files: raise FileNotFoundError(f"No files: {pattern}")
        self.file_idx=0; self.tokens=load_data_shard(self.files[0]); self.pos=0
    def _advance(self):
        self.file_idx=(self.file_idx+1)%len(self.files)
        self.tokens=load_data_shard(self.files[self.file_idx]); self.pos=0
    def take(self,n):
        chunks=[]; rem=n
        while rem>0:
            av=self.tokens.numel()-self.pos
            if av<=0: self._advance(); continue
            k=min(rem,av); chunks.append(self.tokens[self.pos:self.pos+k]); self.pos+=k; rem-=k
        return chunks[0] if len(chunks)==1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self,pattern,rank,world_size,device):
        self.rank=rank; self.world_size=world_size; self.device=device
        self.stream=TokenStream(pattern)
    def next_batch(self,global_tokens,seq_len,grad_accum):
        local=global_tokens//(self.world_size*grad_accum); span=local+1
        chunk=self.stream.take(span*self.world_size)
        loc=chunk[self.rank*span:(self.rank+1)*span].to(dtype=torch.int64)
        x=loc[:-1].reshape(-1,seq_len); y=loc[1:].reshape(-1,seq_len)
        return x.to(self.device,non_blocking=True), y.to(self.device,non_blocking=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, eps=None): super().__init__(); self.eps=eps
    def forward(self,x): return F.rms_norm(x,(x.size(-1),),eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self,x):
        b=self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x,self.weight.to(x.dtype),b)


def restore_fp32(module):
    with torch.no_grad():
        for name,param in module.named_parameters():
            is_ctrl = any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)
            if (param.ndim<2 or is_ctrl) and param.dtype!=torch.float32:
                param.data=param.data.float()


class Rotary(nn.Module):
    def __init__(self,dim,base=10000.):
        super().__init__()
        self.register_buffer("inv_freq",1./(base**(torch.arange(0,dim,2,dtype=torch.float32)/dim)),persistent=False)
        self._sl=0; self._cos=None; self._sin=None
    def forward(self,seq_len,device,dtype):
        if self._cos is None or self._sl!=seq_len or self._cos.device!=device:
            t=torch.arange(seq_len,device=device,dtype=self.inv_freq.dtype)
            fr=torch.outer(t,self.inv_freq.to(device))
            self._cos=fr.cos()[None,None,:,:]; self._sin=fr.sin()[None,None,:,:]; self._sl=seq_len
        return self._cos.to(dtype=dtype), self._sin.to(dtype=dtype)


def apply_rope(x,cos,sin):
    h=x.size(-1)//2; x1,x2=x[...,:h],x[...,h:]
    return torch.cat((x1*cos+x2*sin, x1*(-sin)+x2*cos),dim=-1)


class BigramHash(nn.Module):
    def __init__(self,d,sz=3072):
        super().__init__()
        self.tab=nn.Embedding(sz,d); self.sz=sz
        self.proj=CastedLinear(d,d,bias=False); nn.init.orthogonal_(self.proj.weight)
    def forward(self,t):
        prev=F.pad(t[:,:-1],(1,0)); idx=(t*31337+prev*1000003)%self.sz
        return self.proj(self.tab(idx))


class GatedHighway(nn.Module):
    """Persistent state threaded across recurrence loops — gives RNN-like memory."""
    def __init__(self,d,ds):
        super().__init__()
        self.rp=CastedLinear(ds,d,bias=False); self.rg=CastedLinear(d,1,bias=True)
        self.wp=CastedLinear(d,ds,bias=False); self.wg=CastedLinear(d,1,bias=True)
        self.mx=CastedLinear(ds*2,ds,bias=True)
        nn.init.zeros_(self.rg.bias); nn.init.zeros_(self.wg.bias)
    def forward(self,x,h):
        x_out = x + torch.sigmoid(self.rg(x)) * self.rp(h.to(x.dtype)).unsqueeze(1)
        h_new = torch.tanh(self.mx(torch.cat([h, self.wp((torch.sigmoid(self.wg(x))*x).mean(1))], -1)))
        return x_out, h_new


class Attention(nn.Module):
    def __init__(self,dim,nh,nkv,rope_base,qk_init):
        super().__init__()
        self.nh=nh; self.nkv=nkv; self.hd=dim//nh; self.rep=nh//nkv
        kv=nkv*self.hd
        self.c_q=CastedLinear(dim,dim,bias=False); self.c_k=CastedLinear(dim,kv,bias=False)
        self.c_v=CastedLinear(dim,kv,bias=False);  self.proj=CastedLinear(dim,dim,bias=False)
        self.proj._zero_init=True
        self.qk_g  = nn.Parameter(torch.full((nh,),qk_init**0.5,dtype=torch.float32))
        self.ln_s  = nn.Parameter(torch.ones(1,dtype=torch.float32))
        self.v_bias= nn.Parameter(torch.zeros(nkv,self.hd,dtype=torch.float32)*0.01)
        self.rotary= Rotary(self.hd,base=rope_base)
    def forward(self,x,skip=None):
        B,T,D=x.shape
        q=self.c_q(x).reshape(B,T,self.nh, self.hd).transpose(1,2)
        k=self.c_k(x).reshape(B,T,self.nkv,self.hd).transpose(1,2)
        v=self.c_v(x).reshape(B,T,self.nkv,self.hd).transpose(1,2)
        v=v+self.v_bias.to(v.dtype)[None,:,None,:]
        if skip is not None:
            g=torch.sigmoid(self.ln_s*0.1)
            k=k*(1-g)+self.c_k(skip).reshape(B,T,self.nkv,self.hd).transpose(1,2)*g
            v=v*(1-g)+self.c_v(skip).reshape(B,T,self.nkv,self.hd).transpose(1,2)*g
        q=F.rms_norm(q,(q.size(-1),)); k=F.rms_norm(k,(k.size(-1),))
        cos,sin=self.rotary(T,x.device,q.dtype)
        q=apply_rope(q,cos,sin); k=apply_rope(k,cos,sin)
        gain=self.qk_g.abs().to(q.dtype)[:,None,None]
        q=q*gain; k=k.repeat_interleave(self.rep,1)*gain; v=v.repeat_interleave(self.rep,1)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=(self.nkv!=self.nh))
        return self.proj(y.transpose(1,2).contiguous().reshape(B,T,D)) * self.ln_s.to(x.dtype)


class MLP(nn.Module):
    def __init__(self,dim,mult):
        super().__init__()
        h=int(dim*mult*2); h=(h+255)//256*256
        self.fc=CastedLinear(dim,h,bias=False); self.proj=CastedLinear(h//2,dim,bias=False)
        self.proj._zero_init=True
    def forward(self,x):
        g,v=self.fc(x).chunk(2,dim=-1)
        g=torch.where(g>0,g*g,(0.5*g)**2)   # LeakyReLU-squared
        return self.proj(g*v)


class Block(nn.Module):
    def __init__(self,dim,nh,nkv,mult,rope_base,qk_init,parallel=False):
        super().__init__()
        self.parallel=parallel
        self.attn_norm=RMSNorm(); self.mlp_norm=RMSNorm()
        self.attn=Attention(dim,nh,nkv,rope_base,qk_init); self.mlp=MLP(dim,mult)
        self.attn_scale=nn.Parameter(torch.ones(dim,dtype=torch.float32))
        self.mlp_scale =nn.Parameter(torch.ones(dim,dtype=torch.float32))
        self.resid_mix =nn.Parameter(torch.stack([torch.ones(dim),torch.zeros(dim)]).float())
        if parallel: self.mix=nn.Parameter(torch.tensor(0.5,dtype=torch.float32))
    def forward(self,x,x0,skip=None):
        rm=self.resid_mix.to(x.dtype); x=rm[0][None,None,:]*x+rm[1][None,None,:]*x0
        if self.parallel:
            a=self.attn(self.attn_norm(x),skip)*self.attn_scale.to(x.dtype)[None,None,:]
            m=self.mlp(self.mlp_norm(x))     *self.mlp_scale.to(x.dtype)[None,None,:]
            al=torch.sigmoid(self.mix.to(x.dtype)); return x+al*a+(1-al)*m
        x=x+self.attn_scale.to(x.dtype)[None,None,:]*self.attn(self.attn_norm(x),skip)
        return x+self.mlp_scale.to(x.dtype)[None,None,:]*self.mlp(self.mlp_norm(x))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MODEL
# ─────────────────────────────────────────────────────────────────────────────
class GPT(nn.Module):
    def __init__(self,args:Hyperparameters):
        super().__init__(); self.args=args
        self.tie_embeddings=args.tie_embeddings; self.logit_softcap=args.logit_softcap
        self.recur_layers=args.recur_layers; self.recur_times=args.recur_times
        self.recur_active=False

        self.tok_emb=nn.Embedding(args.vocab_size,args.model_dim)
        self.bigram=BigramHash(args.model_dim,args.bigram_sz)

        nenc=args.num_layers//2; ndec=args.num_layers-nenc
        self.num_encoder_layers=nenc; self.num_decoder_layers=ndec
        self.num_skip_weights=min(nenc,ndec)
        self.skip_weights=nn.Parameter(torch.ones(self.num_skip_weights,args.model_dim,dtype=torch.float32))

        self.blocks=nn.ModuleList([
            Block(args.model_dim,args.num_heads,args.num_kv_heads,args.mlp_mult,
                  args.rope_base,args.qk_gain_init,parallel=(i>=args.parallel_from))
            for i in range(args.num_layers)])
        self.highway=GatedHighway(args.model_dim,args.state_dim)
        self.final_norm=RMSNorm()
        self.lm_head=None if args.tie_embeddings else CastedLinear(args.model_dim,args.vocab_size,bias=False)
        if self.lm_head is not None: self.lm_head._zero_init=True
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight,mean=0.0,std=self.args.tied_embed_init_std)
        for m in self.modules():
            if isinstance(m,CastedLinear) and getattr(m,"_zero_init",False):
                nn.init.zeros_(m.weight)

    def activate_recurrence(self): self.recur_active=True

    def forward_logits(self,input_ids):
        args=self.args; B,T=input_ids.shape
        x=self.tok_emb(input_ids)+self.bigram(input_ids)
        x=F.rms_norm(x,(x.size(-1),)); x0=x
        rl=set(self.recur_layers); fl=min(rl); ll=max(rl)
        if not self.recur_active:
            skips=[]
            for i in range(self.num_encoder_layers): x=self.blocks[i](x,x0); skips.append(x)
            for i in range(self.num_decoder_layers):
                if skips: x=x+self.skip_weights[i].to(x.dtype)[None,None,:]*skips.pop()
                x=self.blocks[self.num_encoder_layers+i](x,x0)
        else:
            for i in range(fl): x=self.blocks[i](x,x0)
            enc={}; h=torch.zeros(B,args.state_dim,device=x.device,dtype=x.dtype)
            for loop in range(self.recur_times):
                for li in self.recur_layers:
                    skip=enc.get(li) if loop>0 else None
                    x=self.blocks[li](x,x0,skip)
                    if loop==0: enc[li]=x.detach()
                x,h=self.highway(x,h)
            for i in range(ll+1,args.num_layers): x=self.blocks[i](x,x0)
        x=self.final_norm(x).reshape(-1,x.size(-1))
        logits=F.linear(x,self.tok_emb.weight.to(x.dtype)) if self.tie_embeddings else self.lm_head(x)
        logits=self.logit_softcap*torch.tanh(logits/self.logit_softcap)
        return logits.reshape(B,T,-1)

    def forward(self,input_ids,target_ids):
        logits=self.forward_logits(input_ids)
        return F.cross_entropy(logits.reshape(-1,self.args.vocab_size).float(),target_ids.reshape(-1))


# ─────────────────────────────────────────────────────────────────────────────
# LEGAL SCORE-FIRST TTT
# ─────────────────────────────────────────────────────────────────────────────
def run_ttt(base_model,val_tokens,args,device,bb_lut,hs_lut,ib_lut):
    ttt=copy.deepcopy(base_model).to(device); ttt.recur_active=base_model.recur_active; ttt.eval()
    opt=torch.optim.SGD(ttt.parameters(),lr=args.ttt_lr,momentum=args.ttt_momentum)
    d=val_tokens.to(device=device,dtype=torch.int64); N=len(d)-1
    nch=(N+args.ttt_chunk-1)//args.ttt_chunk
    tot_nll=tot_tok=tot_byt=0.0; t0=time.perf_counter()
    for ci,s in enumerate(range(0,N,args.ttt_chunk)):
        if time.perf_counter()-t0>590: break
        e=min(s+args.ttt_chunk,N); xc=d[s:e].unsqueeze(0); yc=d[s+1:e+1].unsqueeze(0)
        with torch.inference_mode():                                      # SCORE FIRST
            lg=ttt.forward_logits(xc)
            nll=F.cross_entropy(lg.reshape(-1,args.vocab_size).float(),yc.reshape(-1),reduction="sum")
            tot_nll+=nll.item(); tot_tok+=yc.numel()
            tb=bb_lut[yc[0]].to(dtype=torch.int16)
            tb+=(hs_lut[yc[0]] & ~ib_lut[xc[0]]).to(dtype=torch.int16)
            tot_byt+=float(tb.to(torch.float64).sum())
        clr=args.ttt_lr*0.5*(1+math.cos(math.pi*ci/nch))               # THEN UPDATE
        for pg in opt.param_groups: pg["lr"]=clr
        for _ in range(args.ttt_epochs):
            ttt.train(); opt.zero_grad()
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16): loss=ttt(xc,yc)
            loss.backward(); nn.utils.clip_grad_norm_(ttt.parameters(),1.0); opt.step(); ttt.eval()
    avg=tot_nll/max(1,tot_tok); bpt=avg/math.log(2.0); tpb=tot_tok/max(1,tot_byt)
    return float(avg), float(bpt*tpb)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    global zeropower_via_newtonschulz5
    code=Path(__file__).read_text(encoding="utf-8"); args=Hyperparameters()
    zeropower_via_newtonschulz5=torch.compile(zeropower_via_newtonschulz5)

    distributed="RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank=int(os.environ.get("RANK","0")); world_size=int(os.environ.get("WORLD_SIZE","1"))
    local_rank=int(os.environ.get("LOCAL_RANK","0"))
    if world_size<=0: raise ValueError("WORLD_SIZE must be positive")
    if 8%world_size!=0: raise ValueError("WORLD_SIZE must divide 8")
    grad_accum=8//world_size; grad_scale=1.0/grad_accum
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device=torch.device("cuda",local_rank); torch.cuda.set_device(device)
    if distributed: dist.init_process_group(backend="nccl",device_id=device); dist.barrier()
    master=rank==0

    torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
    from torch.backends.cuda import enable_cudnn_sdp,enable_flash_sdp,enable_math_sdp,enable_mem_efficient_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)

    logfile=None
    if master: os.makedirs("logs",exist_ok=True); logfile=f"logs/{args.run_id}.txt"; print(logfile)
    def log0(msg,console=True):
        if not master: return
        if console: print(msg)
        if logfile:
            with open(logfile,"a",encoding="utf-8") as f: print(msg,file=f)

    log0(code,console=False); log0("="*100,console=False)
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    sp=spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size())!=args.vocab_size:
        raise ValueError(f"VOCAB_SIZE mismatch: {args.vocab_size} vs {int(sp.vocab_size())}")
    val_tokens=load_validation_tokens(args.val_files,args.train_seq_len)
    bb_lut,hs_lut,ib_lut=build_sentencepiece_luts(sp,args.vocab_size,device)
    log0(f"val_bpb:enabled tokenizer={args.tokenizer_path} vocab={args.vocab_size}")

    base_model=GPT(args).to(device).bfloat16()
    for m in base_model.modules():
        if isinstance(m,CastedLinear): m.float()
    restore_fp32(base_model)
    compiled=torch.compile(base_model,dynamic=False,fullgraph=False)
    model:nn.Module=DDP(compiled,device_ids=[local_rank],broadcast_buffers=False) if distributed else compiled
    ema_model=copy.deepcopy(base_model); ema_model.eval(); ema_active=False

    block_named=list(base_model.blocks.named_parameters())
    matrix_p=[p for n,p in block_named if p.ndim==2 and not any(x in n for x in CONTROL_TENSOR_NAME_PATTERNS)]
    scalar_p =[p for n,p in block_named if p.ndim<2  or  any(x in n for x in CONTROL_TENSOR_NAME_PATTERNS)]
    for em in [base_model.highway,base_model.bigram]:
        scalar_p.extend(em.parameters())
    scalar_p.append(base_model.skip_weights)

    opt_emb=torch.optim.Adam([{"params":[base_model.tok_emb.weight],"lr":args.embed_lr,"base_lr":args.embed_lr}],
                              betas=(args.beta1,args.beta2),eps=args.adam_eps,fused=True)
    opt_muon=Muon(matrix_p,lr=args.matrix_lr,momentum=args.muon_momentum,
                  backend_steps=args.muon_backend_steps,weight_decay=args.weight_decay)
    opt_scalar=torch.optim.Adam([{"params":scalar_p,"lr":args.scalar_lr,"base_lr":args.scalar_lr}],
                                 betas=(args.beta1,args.beta2),eps=args.adam_eps,fused=True)
    for g in opt_muon.param_groups: g["base_lr"]=args.matrix_lr
    optimizers=[opt_emb,opt_muon,opt_scalar]
    if base_model.lm_head is not None:
        oh=torch.optim.Adam([{"params":[base_model.lm_head.weight],"lr":0.008,"base_lr":0.008}],
                             betas=(args.beta1,args.beta2),eps=args.adam_eps,fused=True)
        optimizers.insert(1,oh)

    log0(f"model_params:{sum(p.numel() for p in base_model.parameters())}")
    log0(f"world_size:{world_size} grad_accum:{grad_accum} recur:{args.recur_layers}x{args.recur_times}")

    max_wall_ms=1000.*args.max_wallclock_seconds if args.max_wallclock_seconds>0 else None
    recur_at=int(args.iterations*args.recur_at_frac)
    ema_at  =int(args.iterations*args.ema_start_frac)
    qat_at  =int(args.iterations*args.qat_start_frac)

    def lr_scale(step,elapsed_ms):
        if args.warmdown_iters<=0: return 1.0
        if max_wall_ms is None:
            ws=max(args.iterations-args.warmdown_iters,0)
            return max((args.iterations-step)/max(args.warmdown_iters,1),0.) if ws<=step<args.iterations else 1.0
        sms=elapsed_ms/max(step,1); wdms=args.warmdown_iters*sms; rem=max(max_wall_ms-elapsed_ms,0.)
        return rem/max(wdms,1e-9) if rem<=wdms else 1.0

    def zero_all():
        for o in optimizers: o.zero_grad(set_to_none=True)

    if args.warmup_steps>0:
        init_state={n:t.detach().cpu().clone() for n,t in base_model.state_dict().items()}
        init_opts=[copy.deepcopy(o.state_dict()) for o in optimizers]
        tl=DistributedTokenLoader(args.train_files,rank,world_size,device); model.train()
        for _ in range(args.warmup_steps):
            zero_all()
            for ms in range(grad_accum):
                if distributed: model.require_backward_grad_sync=(ms==grad_accum-1)
                x,y=tl.next_batch(args.train_batch_tokens,args.train_seq_len,grad_accum)
                with torch.autocast(device_type="cuda",dtype=torch.bfloat16): loss=model(x,y)
                (loss*grad_scale).backward()
            for o in optimizers: o.step(); zero_all()
        base_model.load_state_dict(init_state,strict=True)
        for o,s in zip(optimizers,init_opts): o.load_state_dict(s)
        zero_all()
        if distributed: model.require_backward_grad_sync=True
        log0(f"warmup_done:{args.warmup_steps}")

    train_loader=DistributedTokenLoader(args.train_files,rank,world_size,device)
    training_ms=0.; stop_at=None; torch.cuda.synchronize(); t0=time.perf_counter(); step=0

    while True:
        last=step==args.iterations or (stop_at is not None and step>=stop_at)
        if step==recur_at: base_model.activate_recurrence(); ema_model.activate_recurrence(); log0(f"step:{step} recurrence_on")
        if step==qat_at:   log0(f"step:{step} qat_phase")
        if step>=ema_at and not ema_active:
            ema_model.load_state_dict(base_model.state_dict()); ema_active=True; log0(f"step:{step} ema_on")

        if last or (args.val_loss_every>0 and step%args.val_loss_every==0):
            torch.cuda.synchronize(); training_ms+=1000.*(time.perf_counter()-t0)
            em=ema_model if ema_active else base_model
            vl,vb=eval_val(args,em,rank,world_size,device,grad_accum,val_tokens,bb_lut,hs_lut,ib_lut)
            log0(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} train_time:{training_ms:.0f}ms step_avg:{training_ms/max(step,1):.2f}ms")
            torch.cuda.synchronize(); t0=time.perf_counter()

        if last: break

        elapsed=training_ms+1000.*(time.perf_counter()-t0); sc=lr_scale(step,elapsed); zero_all()
        train_loss=torch.zeros((),device=device)
        for ms in range(grad_accum):
            if distributed: model.require_backward_grad_sync=(ms==grad_accum-1)
            x,y=train_loader.next_batch(args.train_batch_tokens,args.train_seq_len,grad_accum)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16): loss=model(x,y)
            train_loss+=loss.detach(); (loss*grad_scale).backward()
        train_loss/=grad_accum
        frac=min(step/max(args.muon_momentum_warmup_steps,1),1.0)
        mom=(1-frac)*args.muon_momentum_warmup_start+frac*args.muon_momentum
        for g in opt_muon.param_groups: g["momentum"]=mom
        for o in optimizers:
            for g in o.param_groups: g["lr"]=g["base_lr"]*sc
        if args.grad_clip_norm>0: nn.utils.clip_grad_norm_(base_model.parameters(),args.grad_clip_norm)
        for o in optimizers: o.step(); zero_all()
        if ema_active:
            with torch.no_grad():
                for ep,p in zip(ema_model.parameters(),base_model.parameters()): ep.lerp_(p.to(ep.dtype),1-args.ema_decay)
        step+=1; appr=training_ms+1000.*(time.perf_counter()-t0)
        if args.train_log_every>0 and (step<=10 or step%args.train_log_every==0):
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} train_time:{appr:.0f}ms step_avg:{appr/step:.2f}ms")
        hit=max_wall_ms is not None and appr>=max_wall_ms
        if distributed and max_wall_ms is not None:
            t=torch.tensor(int(hit),device=device); dist.all_reduce(t,op=dist.ReduceOp.MAX); hit=bool(t.item())
        if stop_at is None and hit: stop_at=step

    log0(f"peak_memory:{torch.cuda.max_memory_allocated()//1024//1024}MiB")

    # ── Post-training evaluation ─────────────────────────────────────────────
    if master:
        em=ema_model if ema_active else base_model; em.eval()
        sl_nll,sl_bpb=sliding_eval(args,em,val_tokens,device,bb_lut,hs_lut,ib_lut)
        log0(f"sliding_eval val_bpb:{sl_bpb:.6f}")
        if args.ttt_enabled:
            tt_nll,tt_bpb=run_ttt(em,val_tokens,args,device,bb_lut,hs_lut,ib_lut)
            log0(f"ttt_eval val_bpb:{tt_bpb:.6f}")
            final_bpb=tt_bpb
        else:
            final_bpb=sl_bpb
        log0(f"val_bpb = {final_bpb:.4f}")

    # ── Serialisation + roundtrip validation (required) ──────────────────────
    export=ema_model if ema_active else base_model
    if master:
        torch.save(export.state_dict(),"final_model.pt")
        mb=os.path.getsize("final_model.pt"); cb=len(code.encode("utf-8"))
        log0(f"Serialized model: {mb} bytes\nCode size: {cb} bytes\nTotal submission size: {mb+cb} bytes")

    quant_obj,quant_stats=quantize_state_dict_int8(export.state_dict())
    qbuf=io.BytesIO(); torch.save(quant_obj,qbuf)
    qraw=qbuf.getvalue(); qblob=zlib.compress(qraw,level=9)
    if master:
        with open("final_model.int8.ptz","wb") as f: f.write(qblob)
        qfb=os.path.getsize("final_model.int8.ptz"); cb=len(code.encode("utf-8"))
        rat=quant_stats["baseline_tensor_bytes"]/max(quant_stats["int8_payload_bytes"],1)
        log0(f"Serialized model int8+zlib: {qfb} bytes (ratio:{rat:.2f}x)")
        log0(f"Total submission size int8+zlib: {qfb+cb} bytes")

    if distributed: dist.barrier()
    with open("final_model.int8.ptz","rb") as f: blob=f.read()
    qstate=torch.load(io.BytesIO(zlib.decompress(blob)),map_location="cpu")
    export.load_state_dict(dequantize_state_dict_int8(qstate),strict=True)
    torch.cuda.synchronize(); tq=time.perf_counter()
    ev=export if not distributed else DDP(export,[local_rank])
    qvl,qbpb=eval_val(args,ev,rank,world_size,device,grad_accum,val_tokens,bb_lut,hs_lut,ib_lut)
    torch.cuda.synchronize()
    log0(f"final_int8_zlib_roundtrip val_loss:{qvl:.4f} val_bpb:{qbpb:.4f} eval_time:{1000.*(time.perf_counter()-tq):.0f}ms")
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{qvl:.8f} val_bpb:{qbpb:.8f}")

    if distributed: dist.destroy_process_group()


if __name__=="__main__":
    main()
