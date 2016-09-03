t = zeros(256);
for i = 1:256
    h = i;
    temp = ones(h, h);
    tic;
    for j = 1:1000
        fft2(temp);
    end
    toc;
    t(i+1)=toc;
end