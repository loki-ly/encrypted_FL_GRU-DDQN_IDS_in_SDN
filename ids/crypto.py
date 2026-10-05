"""Encrypted aggregation backends + packet authentication.

Efficiency design (differs from the reference papers on purpose):
  * clients encrypt the *delta* (w_local - w_global) already multiplied by its FedAvg weight n_k/N,
    so the server needs ADDITIONS ONLY -> multiplicative depth 0 -> smallest CKKS modulus chain,
    no relinearisation/rescale, lowest noise.
  * only the online Q-network is exchanged (the target net is re-synced locally) -> half the traffic of
    aggregating online+target networks.
Backends
  none     plaintext (baseline)
  tenseal  CKKS via TenSEAL, key pair from a trusted setup (Paper 2 style: server never holds sk)
  openfhe  CKKS via OpenFHE *multiparty* keygen: no dealer, joint public key, n-of-n collaborative
           decryption (Paper 3 style; note: the t-of-n Shamir layer of Paper 3 is custom C++ and not included)
"""
import hashlib
import hmac
import os
import time

import numpy as np


class Plain:
    name = "none"

    def setup(self, n_clients):
        return 0.0

    def encrypt(self, cid, vec):
        return vec.astype(np.float32).copy(), vec.nbytes // 1

    def aggregate(self, cts):
        return np.sum(cts, 0)

    def decrypt(self, ct, n):
        return ct[:n].astype(np.float32)

    def blob(self, ct):
        return ct.tobytes()


class TenSEALBackend:
    name = "tenseal"

    def __init__(self, poly=8192, bits=(60, 60), scale_bits=40):
        import tenseal as ts
        self.ts, self.poly, self.bits, self.sb = ts, poly, list(bits), scale_bits
        self.slots = poly // 2

    def setup(self, n_clients):
        t = time.time()
        ts = self.ts
        self.ctx = ts.context(ts.SCHEME_TYPE.CKKS, self.poly, coeff_mod_bit_sizes=self.bits)
        self.ctx.global_scale = 2 ** self.sb
        self.pub = ts.context_from(self.ctx.serialize(save_secret_key=False))   # server: no secret key
        return time.time() - t

    def encrypt(self, cid, vec):
        chunks = [vec[i:i + self.slots] for i in range(0, len(vec), self.slots)]
        cts = [self.ts.ckks_vector(self.ctx, c.astype(np.float64).tolist()).serialize() for c in chunks]
        return cts, sum(len(c) for c in cts)

    def aggregate(self, all_cts):                       # server side: deserialize with PUBLIC ctx, add
        out = None
        for cts in all_cts:
            vs = [self.ts.ckks_vector_from(self.pub, b) for b in cts]
            out = vs if out is None else [a + b for a, b in zip(out, vs)]
        return [v.serialize() for v in out]

    def decrypt(self, cts, n):
        dec = [np.array(self.ts.ckks_vector_from(self.ctx, b).decrypt()) for b in cts]
        return np.concatenate(dec)[:n].astype(np.float32)

    def blob(self, ct):
        return b"".join(ct)


class OpenFHEBackend:
    name = "openfhe"

    def __init__(self, ring=16384, scale_bits=40, first_bits=60):
        import openfhe as of
        self.of, self.ring, self.sb, self.fb = of, ring, scale_bits, first_bits
        self.slots = ring // 2

    def setup(self, n_clients):
        of, t = self.of, time.time()
        p = of.CCParamsCKKSRNS()
        p.SetMultiplicativeDepth(0); p.SetScalingModSize(self.sb); p.SetFirstModSize(self.fb)
        p.SetRingDim(self.ring); p.SetBatchSize(self.slots)
        p.SetSecurityLevel(of.SecurityLevel.HEStd_128_classic)
        self.cc = of.GenCryptoContext(p)
        for f in (of.PKESchemeFeature.PKE, of.PKESchemeFeature.KEYSWITCH, of.PKESchemeFeature.LEVELEDSHE,
                  of.PKESchemeFeature.ADVANCEDSHE, of.PKESchemeFeature.MULTIPARTY):
            self.cc.Enable(f)
        # multiparty key generation: client i extends the joint public key; nobody knows the joint secret
        self.kps = [self.cc.KeyGen()]
        for _ in range(1, n_clients):
            self.kps.append(self.cc.MultipartyKeyGen(self.kps[-1].publicKey))
        self.pk = self.kps[-1].publicKey
        return time.time() - t

    def encrypt(self, cid, vec):
        cts, nb = [], 0
        for i in range(0, len(vec), self.slots):
            pt = self.cc.MakeCKKSPackedPlaintext(vec[i:i + self.slots].astype(np.float64).tolist())
            ct = self.cc.Encrypt(self.pk, pt)
            cts.append(ct)
            nb += len(self.of.Serialize(ct, self.of.BINARY))
        return cts, nb

    def aggregate(self, all_cts):
        out = list(all_cts[0])
        for cts in all_cts[1:]:
            out = [self.cc.EvalAdd(a, b) for a, b in zip(out, cts)]
        return out

    def decrypt(self, cts, n):                          # collaborative decryption (lead + all mains, then fusion)
        vals = []
        for ct in cts:
            lead = self.cc.MultipartyDecryptLead([ct], self.kps[0].secretKey)
            mains = [self.cc.MultipartyDecryptMain([ct], k.secretKey) for k in self.kps[1:]]
            pt = self.cc.MultipartyDecryptFusion([lead[0]] + [m[0] for m in mains])
            pt.SetLength(self.slots)
            vals.append(np.array(pt.GetRealPackedValue()))
        return np.concatenate(vals)[:n].astype(np.float32)

    def blob(self, ct):
        return b"".join(self.of.Serialize(c, self.of.BINARY) for c in ct)


def make_backend(name):
    return {"none": Plain, "tenseal": TenSEALBackend, "openfhe": OpenFHEBackend}[name]()


# ------------------------------------------------------------------ authenticity / integrity / replay
class Registry:
    """Trusted registration issues one HMAC key per client. Server verifies hash + tag + freshness + nonce."""

    def __init__(self, n, window_s=60.0):
        self.keys = {i: os.urandom(32) for i in range(n)}
        self.window, self.seen = window_s, set()

    def sign(self, cid, rnd, blob):
        ts, nonce = time.time(), os.urandom(8).hex()
        h = hashlib.sha256(blob).hexdigest()
        tag = hmac.new(self.keys[cid], f"{cid}|{rnd}|{ts}|{nonce}|{h}".encode(), hashlib.sha256).hexdigest()
        return {"cid": cid, "round": rnd, "ts": ts, "nonce": nonce, "hash": h, "tag": tag}

    def verify(self, pkt, blob):
        if pkt["cid"] not in self.keys or abs(time.time() - pkt["ts"]) > self.window:
            return False
        if pkt["nonce"] in self.seen or hashlib.sha256(blob).hexdigest() != pkt["hash"]:
            return False
        exp = hmac.new(self.keys[pkt["cid"]], f"{pkt['cid']}|{pkt['round']}|{pkt['ts']}|{pkt['nonce']}|{pkt['hash']}"
                       .encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(exp, pkt["tag"]):
            return False
        self.seen.add(pkt["nonce"])
        return True
