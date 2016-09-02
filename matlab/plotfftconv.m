tic;O1 = conv2(B,S);toc;
tic;O2 = conv2fft(B,S);toc;
tic;O3 = conv2fftBlock(B,S);toc;
subplot(2,2,1);
imshow(O1, [])
title('original')
subplot(2,2,2);
imshow(O2, [])
title('fft')
subplot(2,2,3);
imshow(O3, [])
title('fftblock')