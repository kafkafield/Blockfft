function [ O ] = conv2fft( I, K )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    [kh ~] = size(K);
    [ih ~] = size(I);
    kpad = padarray(K, size(I) - size(K), 'post');
    %kpad = zeros(ih, ih);
    %kpad(1:ih, 1:ih) = I;
    o2 = ifft2(fft2(I).*fft2(kpad));
    sizeO = size(I) - size(K);
    O = o2(kh:kh+sizeO(1), kh:kh+sizeO(2));
end

