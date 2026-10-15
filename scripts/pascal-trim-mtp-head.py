#!/usr/bin/env python3
# Pascal fork (H45): copy a Qwen3.5/3.8 MTP drafter GGUF and add
# blk.<N>.nextn.shared_head_head.weight = the first K rows of output.weight.
# The MTP graph prefers that tensor, so the drafter only scores token ids < K.
# The target still verifies every draft token, so output is unchanged.
# Only acceptance can drop, by about the share of target tokens with id >= K.
# usage: pascal-trim-mtp-head.py in.gguf out.gguf K
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'gguf-py'))
import gguf  # noqa: E402

src, dst, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
r = gguf.GGUFReader(src)
arch = r.fields[gguf.Keys.General.ARCHITECTURE].contents()
n_layer = r.fields[f'{arch}.block_count'].contents()
head_name = f'blk.{n_layer - 1}.nextn.shared_head_head.weight'
assert not any(t.name == head_name for t in r.tensors), f'{head_name} already present'

out = next(t for t in r.tensors if t.name == 'output.weight')
assert 0 < k <= out.data.shape[0], f'K={k} outside [1, {out.data.shape[0]}]'
head = out.data[:k]  # rows are the vocab axis; quant blocks run along a row, so a row prefix is a valid tensor

w = gguf.GGUFWriter(dst, arch=arch, endianess=r.endianess)
for f in r.fields.values():
    if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith('GGUF.'):
        continue
    vt = f.types[0]
    w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)

tensors = [(t.name, t.data, t.tensor_type) for t in r.tensors] + [(head_name, head, out.tensor_type)]
for name, data, qt in tensors:
    w.add_tensor_info(name, data.shape, data.dtype, data.nbytes, qt)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()
for _, data, _ in tensors:
    w.write_tensor_data(data, tensor_endianess=r.endianess)
w.close()
print(f'{dst}: {head_name} {out.tensor_type.name} [{out.data.shape[1]} bytes x {k} rows]')
