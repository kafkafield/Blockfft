t = ones(11);
for i = 0:10
    h = 2^i;
    temp = ones(h, h);
    tic;
    for j = 1:1
        fft2(temp);
    end
    toc;
    t(i+1)=toc;
end