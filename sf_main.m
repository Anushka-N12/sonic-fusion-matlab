close all;
clear;
clc;

filepath = "attention.wav";
[y,Fs_orig] = audioread(filepath);  % Use a different variable to store the original Fs

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

% Generate the spectrogram
% Arguement 1: Hamming window function smoothens the movement from one
% segment to another
% figure(1)
% spectrogram(y, hamming(samples_per_segment), overlap, samples_per_segment, Fs_orig, 'yaxis');
% title('Spectrogram of attention.wav');
% disp(s);

% Calculate the mixture spectrogram
[S, F, T] = spectrogram(y, hamming(samples_per_segment), overlap, samples_per_segment, Fs_orig, 'yaxis');

% Display the spectrogram
figure;
imagesc(T, F, 10*log10(abs(S))); % Plot the spectrogram in dB scale
axis xy; % Flip the y-axis direction to match the conventional axis orientation
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Spectrogram of attention.wav');


