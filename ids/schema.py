"""Canonical CICFlowMeter schema so InSDN / CICIDS2017 / CICDDoS2019 share one feature space.

InSDN uses short names ("Tot Fwd Pkts"), CICIDS2017/CICDDoS2019 use long names
("Total Fwd Packets"). Both are mapped to the same canonical key via a normalised name.
"""
import re


def nz(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


A = {"flow_duration": ["flowduration"],
     "tot_fwd_pkts": ["totalfwdpackets", "totfwdpkts"],
     "tot_bwd_pkts": ["totalbackwardpackets", "totbwdpkts"],
     "flow_byts_s": ["flowbytess", "flowbytss"],
     "flow_pkts_s": ["flowpacketss", "flowpktss"],
     "down_up_ratio": ["downupratio"],
     "pkt_size_avg": ["averagepacketsize", "pktsizeavg"],
     "pkt_len_min": ["minpacketlength", "pktlenmin"],
     "pkt_len_max": ["maxpacketlength", "pktlenmax"],
     "pkt_len_mean": ["packetlengthmean", "pktlenmean"],
     "pkt_len_std": ["packetlengthstd", "pktlenstd"],
     "pkt_len_var": ["packetlengthvariance", "pktlenvar"],
     "init_fwd_win": ["initwinbytesforward", "initfwdwinbyts"],
     "init_bwd_win": ["initwinbytesbackward", "initbwdwinbyts"],
     "fwd_act_data_pkts": ["actdatapktfwd", "fwdactdatapkts"],
     "fwd_seg_size_min": ["minsegsizeforward", "fwdsegsizemin"],
     "protocol": ["protocol"]}
for d in ("fwd", "bwd"):
    A[f"totlen_{d}_pkts"] = [f"totallengthof{d}packets", f"totlen{d}pkts"]
    for s in ("max", "min", "mean", "std"):
        A[f"{d}_pkt_len_{s}"] = [f"{d}packetlength{s}", f"{d}pktlen{s}"]
        A[f"{d}_iat_{s}"] = [f"{d}iat{s}"]
    A[f"{d}_iat_tot"] = [f"{d}iattotal", f"{d}iattot"]
    A[f"{d}_psh_flags"] = [f"{d}pshflags"]
    A[f"{d}_urg_flags"] = [f"{d}urgflags"]
    A[f"{d}_header_len"] = [f"{d}headerlength", f"{d}headerlen"]
    A[f"{d}_pkts_s"] = [f"{d}packetss", f"{d}pktss"]
    A[f"{d}_seg_size_avg"] = [f"avg{d}segmentsize", f"{d}segsizeavg"]
    A[f"subflow_{d}_pkts"] = [f"subflow{d}packets", f"subflow{d}pkts"]
    A[f"subflow_{d}_byts"] = [f"subflow{d}bytes", f"subflow{d}byts"]
for s in ("mean", "std", "max", "min"):
    A[f"flow_iat_{s}"] = [f"flowiat{s}"]
    A[f"active_{s}"] = [f"active{s}"]
    A[f"idle_{s}"] = [f"idle{s}"]
for f in ("fin", "syn", "rst", "psh", "ack", "urg", "cwe", "ece"):
    A[f"{f}_flag_cnt"] = [f"{f}flagcount", f"{f}flagcnt"]

FEATURES = list(A)
META = {"timestamp": ["timestamp"], "dst_ip": ["destinationip", "dstip"],
        "src_port": ["sourceport", "srcport"], "dst_port": ["destinationport", "dstport"],
        "label": ["label"]}
REV = {a: k for k, v in {**A, **META}.items() for a in v}


def canonicalize(df):
    """Rename columns to canonical keys; keep only features + meta columns."""
    cols = {}
    for c in df.columns:
        k = REV.get(nz(c))
        if k and k not in cols:
            cols[k] = c
    out = df[list(cols.values())].copy()
    out.columns = list(cols.keys())
    return out
