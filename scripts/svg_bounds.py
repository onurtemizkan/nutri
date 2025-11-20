#!/usr/bin/env python3
import re
from pathlib import Path

svg = Path('logo.svg').read_text()
# collect all d="..." contents
ds = re.findall(r'd="([^"]+)"', svg)
nums = []
for d in ds:
    # find all numbers (including negatives and decimals)
    found = re.findall(r'[-+]?[0-9]*\.?[0-9]+', d)
    nums.extend(found)
# convert to floats
fs = [float(x) for x in nums]
# group sequential pairs into x,y
xs = []
ys = []
for i in range(0, len(fs)-1, 2):
    x = fs[i]
    y = fs[i+1]
    xs.append(x)
    ys.append(y)
if not xs:
    print('No coords found')
    raise SystemExit(1)
minx = min(xs)
maxx = max(xs)
miny = min(ys)
maxy = max(ys)
print(f'minx={minx}\nmaxx={maxx}\nminy={miny}\nmaxy={maxy}')
# original canvas from viewBox
vb = re.search(r'viewBox="([^"]+)"', svg)
if vb:
    v = [float(x) for x in vb.group(1).split()]  # minx miny w h
    orig_x, orig_y, orig_w, orig_h = v
else:
    orig_x, orig_y, orig_w, orig_h = 0.0,0.0,1024.0,1024.0
pad_left = minx - orig_x
pad_right = (orig_x+orig_w) - maxx
pad_top = miny - orig_y
pad_bottom = (orig_y+orig_h) - maxy
print(f'pad_left={pad_left}\npad_right={pad_right}\npad_top={pad_top}\npad_bottom={pad_bottom}')
# reduce padding by factor 3
new_minx = minx - pad_left/3.0
new_miny = miny - pad_top/3.0
new_maxx = maxx + pad_right/3.0
new_maxy = maxy + pad_bottom/3.0
new_w = new_maxx - new_minx
new_h = new_maxy - new_miny
print(f'new_viewBox: {new_minx} {new_miny} {new_w} {new_h}')
# compute new width/height preserving scale factor of original (we used 1.8 earlier assuming 1024)
scale = 1.8
new_width = new_w * scale
new_height = new_h * scale
print(f'new_width={new_width}\nnew_height={new_height}')
