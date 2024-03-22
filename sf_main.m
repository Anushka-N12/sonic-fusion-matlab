close all;
clear;
clc;

disp('Starting program');
filepath = "The Good Soldier (mp3cut.net).wav";
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
title('Mixed Spectrogram of audio file');

S_size = size(S);
disp(S_size);

% Slide the rows of the mixture spectrogram
% Direct call didn't work due to memory limits

disp('Running autocorr loop...');
tic;

% Calculate autocorrelation for first row
corr_r = xcorr(S(1, :));
B = zeros(size(corr_r, 2), S_size(1));  % Initialize the autocorrelation matrix
B(:, 1) = corr_r;

% Calculate autocorrelation for each row
for i = 2:S_size(1)
    corr_r = xcorr(S(i, :));
    B(:, i) = corr_r;
end
et = datestr(datenum(0,0,0,0,0,toc), 'MM:SS');
disp(['Time taken to run autocorr loop - ', et]);

% Compute the mean value for each row of matrix B to obtain the beat spectrum b
bs_r = mean(B, 2); % Compute mean along the rows (dimension 2)

% Normalisation using first value
bs_r_n = bs_r / bs_r(1);

t = 1:size(bs_r_n);

% Plot the final beat spectrum
figure;
plot(t, bs_r_n);
xlabel('Index');
ylabel('Value');
title('Final Beat Spectrum');




