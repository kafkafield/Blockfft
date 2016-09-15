import pylab as plt
import numpy as np
import re

debug = True

f_reg = re.compile('^Forward(\d+)\:(\d+\.\d+)')
b_reg = re.compile('^Backward(\d+)\:(\d+\.\d+)')

f = open('fft_runtime.log', 'r')
fx = []
bx = []
ft = []
bt = []

while(1):
    buffer = f.readline()
    if not buffer:
        break
    match1 = f_reg.match(buffer)
    if match1:
        tlist = match1.groups()
        if debug:
            print tlist
        fx.append(int(tlist[0]))
        ft.append(float(tlist[1]))
    match2 = b_reg.match(buffer)
    if match2:
        tlist = match2.groups()
        if debug:
            print tlist
        bx.append(int(tlist[0]))
        bt.append(float(tlist[1]))

f.close()

if debug:
    print fx
    print ft
    print bx
    print bt

fig, (ax1, ax2) = plt.subplots(2, sharex=True)

ax1.plot(fx, ft, 'ob-', label='fft')

ax1.xlabel('Input Shape')
ax1.ylabel('Run time(s)')

ax1.legend()

ax2.plot(bx, bt, 'pr-', label='ifft')

ax2.xlabel('Input Shape')
ax2.ylabel('Run time(s)')

ax2.legend()
plt.show()
