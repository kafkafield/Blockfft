function [ O ] = conv2fftBlock( I, K )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    [ih, ~] = size(I);
    [kh, ~] = size(K);
    n = ceil(ih / kh);
    %ipad = padarray(I, size(K)*n-size(I), 'post');
    %ipad = padarray(ipad, floor(size(K)/2));
    ipad = zeros(n*kh+kh-1, n*kh+kh-1);
    ipad((kh-1)/2+1:(kh-1)/2+ih, (kh-1)/2+1:(kh-1)/2+ih) = I;
    kpad = padarray(K, size(K)-1, 'post');
    fftK = fft2(kpad);
    O = zeros(size(K)*n);
    col = zeros([kh n*kh]);
    for i = 1:n
        for j = 1:n
            %tempi = imcrop(ipad, [(i-1)*kh+1 (j-1)*kh+1 2*kh-2 2*kh-2]);
            tempi = ipad((i-1)*kh+1:(i-1)*kh+1+2*kh-2, (j-1)*kh+1:(j-1)*kh+1+2*kh-2);
            tempo = ifft2(fft2(tempi).*fftK);
            %tempo = conv2(tempi, K, 'valid');
            %tempo = imcrop(tempo, [kh kh kh-1 kh-1]);
            tempo = tempo(kh:2*kh-1, kh:2*kh-1);
            %col = cat(1, col, tempo);
            col(1:kh, (j-1)*kh+1:j*kh)=tempo;
        end
        %O = cat(2, O, col);
        O((i-1)*kh+1:i*kh, 1:kh*n) = col;
    end
    
    %O = imcrop(o2, [3 3 sizeO2(1) sizeO(2)]);
end

