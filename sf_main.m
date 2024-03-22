close all;
clear;
clc;

disp('Starting program');
filepath = "attention.wav";
[y,Fs_orig] = audioread(filepath);  % Use a different variable to store the original Fs
disp('Loaded .wav file');

% To remove error saying input signal is not a vector
if size(y, 2) > 1
    y = y(:, 1); % Take the first channel if it's a stereo signal
end

time_interval = 0.04;  % Time interval in seconds
samples_per_segment = 2048;
Fs = 44100;  % Sampling frequency

% Calculate the number of samples corresponding to the specified time interval
samples_per_interval = round(time_interval * Fs);

% Determine the overlap based on the desired samples per segment
overlap = samples_per_segment - samples_per_interval;

% Calculate the mixture spectrogram
[S, F, T] = spectrogram(y, hamming(samples_per_segment), overlap, samples_per_segment, Fs_orig, 'yaxis');
disp('Spectrogram done');

% Display the spectrogram
figure;
imagesc(T, F, 10*log10(abs(S))); % Plot the spectrogram in dB scale
axis xy; % Flip the y-axis direction to match the conventional axis orientation
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Spectrogram of attention.wav');

S_size = size(S);
disp(S_size);

% Slide the rows of the mixture spectrogram
% Direct call didn't work due to memory limits
B = zeros(S_size);  % Initialize the autocorrelation matrix

disp('Running autocorr loop...');
tic;
% Calculate autocorrelation for each column
for i = 1:S_size(2)
    lagged_c = circshift(S(:, i), 1); % Shift by one sample to create a lagged version

    corr_c = xcorr(S(:, i), lagged_c);  % This will compute the autocorrelation of each row
    B(i, :) = corr_c(1, floor(length(corr_c(1,:))/2)+1:end); % Take the second half only
end
et = datestr(datenum(0,0,0,0,0,toc), 'MM:SS');
disp(['Time taken to run autocorr loop - ', et]);

% Compute the mean value for each row/column of matrix B to obtain the beat spectrum b
bs_r = mean(B, 2); % Compute mean along the rows (dimension 2)

% Normalisation
bs_r_n = normalize(bs_r);

% Plot the final beat spectrum
figure;
plot(bs_r);
xlabel('Index');
ylabel('Value');
title('Final Beat Spectrum');


