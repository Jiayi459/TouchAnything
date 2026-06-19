"""Model factory."""
from .conv_rnn import ConvRNNSeq2Seq
from .simvp import SimVP


def build_model(cfg):
    name = cfg["name"].lower()
    in_ch = cfg.get("in_ch", 2)
    t_in = cfg.get("t_in", 10)
    t_out = cfg.get("t_out", 15)
    if name in ("convgru", "convlstm"):
        cell = "gru" if name == "convgru" else "lstm"
        return ConvRNNSeq2Seq(cell=cell, in_ch=in_ch,
                              hid=cfg.get("hid", 64), layers=cfg.get("layers", 2),
                              k=cfg.get("kernel", 3), t_out=t_out)
    if name == "simvp":
        return SimVP(in_ch=in_ch, t_in=t_in, t_out=t_out, hid=cfg.get("hid", 64),
                     n_enc=cfg.get("n_enc", 2), n_trans=cfg.get("n_trans", 4),
                     k=cfg.get("kernel", 3))
    raise ValueError(f"unknown model {name!r}")
