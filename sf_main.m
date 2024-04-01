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

disp('Original signal size');
disp(size(y));

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
disp('Size of final beat spectrum is');
bs_size = size(bs_r_n);
disp(bs_size);
bs_real = real(bs_r_n);

% Plot the final beat spectrum
t = 1:bs_size;
figure;
plot(t, bs_real);
xlabel('Index');
ylabel('Value');
title('Final Beat Spectrum');
hold on;

% Find peaks in the beat spectrum
[peaks, locs] = findpeaks(bs_real);
disp("No. of local peaks found is");
disp(size(peaks));

% Find which specific peaks are occuring periodically
% Method not specified in paper
% Making an assumption that all clips will be above 5s
dur = length(y)/Fs; % total duration of audio signal in s
x = 5;
s_per_xs = round(bs_size(1)/dur*x);  % No. of rows in x s
size_ref = bs_size(1);
repeating_peaks = zeros(bs_size(1)+1);
repeating_locs = zeros(bs_size(1)+1);
loc_repeated_at = zeros(bs_size(1)+1);
allocated_till = 1;
for i = 1:size(peaks)
    peak = peaks(i);
    loc = locs(i);
    limit = loc + s_per_xs;
    if ~ismember(peak, repeating_peaks)
        p_size = size(peaks);
        if p_size(1) == j
            break
        end
        j = i+1;
        while (locs(j) < limit) && (j < p_size(1))
            s1 = num2str(peak);
            s2 = num2str(peaks(j));
            if s1(1) == '-'
                s1 = s1(2:end);
            end
            if s2(1) == '-'
                s2 = s2(2:end);
            end

            if size(s1,2) == size(s2,2) && strcmp(s1(1,1:floor(size(s1,2)/4)), s2(1,1:floor(size(s2,2)/4)))
                repeating_peaks(allocated_till) = peak;
                repeating_locs(allocated_till) = loc;
                loc_repeated_at(allocated_till) = locs(j);
                allocated_till = allocated_till + 1;
                break
            end
            j = j + 1;
        end
    end
end
all_zeros = find(repeating_peaks==0); 
first_zero = all_zeros(1)-1;
repeating_peaks = repeating_peaks(1:first_zero);
repeating_locs = repeating_locs(1:first_zero);
loc_repeated_at = loc_repeated_at(1:first_zero);

disp("Size of repeating peaks is");
disp(size(repeating_peaks));

disp("Repeating peaks");
disp(repeating_peaks);
disp("Repeating locs");
disp(repeating_locs);
disp("Above peaks repeated at locs below");
disp(loc_repeated_at);

for i = 1:numel(repeating_locs)
    loc = repeating_locs(i);
    peak = repeating_peaks(i);
    plot(loc, peak, 'ro', 'MarkerSize', 5); % Red circles as markers
end

hold off; % Release the hold on the plot

% Calculate the time difference between consecutive peaks
% Dealing with first peak only for now
k = 1;
repeating_c = loc_repeated_at(k)-repeating_locs(k);
disp("Repeating set of columns");
disp(repeating_c);
% periods = diff(locs);
bs_per_1s = round(bs_size(2)/dur);  % No. of columns in 1s
segment_length = repeating_c;
% Calculate the number of segments based on the repeating period
num_segments = floor(S_size(2) / segment_length);
disp("Number is segments being created is");
disp(num_segments);

% Initialize segmented spectrogram matrix
% Note the matlab definition is like M(x,y,z), unlike numpy M(z,x,y)
segmented_S = zeros(S_size(1), segment_length, num_segments);
disp('Size of segmented_S is');
disp(size(segmented_S));

% Segment the mixture spectrogram
for i = 1:num_segments
    start_index = (i - 1) * segment_length + 1;
    end_index = start_index + segment_length - 1;
    segment = S(:, start_index:end_index);
    segmented_S(:, :, i) = segment;
end

repeated_seg = median(segmented_S, 3);
disp('A segment is of size');
disp(size(repeated_seg));

repeating_S = zeros(S_size);
for i = 1:num_segments
    start_index = (i - 1) * segment_length + 1;
    end_index = start_index + segment_length - 1;
    repeating_S(:, start_index:end_index) = min(segmented_S(:,:,i), repeated_seg);
end

% Display the spectrogram
figure;
imagesc(T, F, 10*log10(abs(repeating_S))); % Plot the spectrogram in dB scale
axis xy; % Flip the y-axis direction to match the conventional axis orientation
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Mixed Spectrogram of audio file');

% Calculate the time-frequency mask
mask = repeating_S ./ S;
disp('Mask shape');
disp(size(mask));

% Normalize the mask values to range from 0 to 1
% Find the maximum and minimum values of the mask array
max_val = max(mask(:));
min_val = min(mask(:));
% Normalize using above
mask = (mask - min_val) / (max_val - min_val);
% Clip the mask values to ensure they are within the range [0, 1]
mask(mask < 0) = 0;
mask(mask > 1) = 1;

% Apply the mask to the original mixture spectrogram
music_spectrogram = mask .* S;
disp("Final spectrogram size");
disp(size(music_spectrogram));

% Check for NaN and Infinite values
nan_indices = isnan(music_spectrogram);
inf_indices = isinf(music_spectrogram);

% Replace NaN and Infinite values with zeros
music_spectrogram(nan_indices) = 0;
music_spectrogram(inf_indices) = 0;

% Display the spectrogram
figure;
imagesc(T, F, 10*log10(abs(repeating_S))); % Plot the spectrogram in dB scale
axis xy; % Flip the y-axis direction to match the conventional axis orientation
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Final music spectrogram');

% Inverse Short-Time Fourier Transform (ISTFT) to get the music signal
music_signal = istft(music_spectrogram, 'Window', hamming(samples_per_segment), 'OverlapLength', overlap, 'FFTLength', samples_per_segment, 'FrequencyRange', 'onesided');
disp('Size of final music signal');
disp(size(music_signal));

%  rest of the reconstruction and audio file writing code
music_signal_n = music_signal / max(abs(music_signal)); % Normalize the signal

% Save Music Signal
music_filepath = "C:\Users\anush\OneDrive\Documents\MATLAB\sonic_fusion\music_signal.wav"; % Choose a file name and path
audiowrite(music_filepath, real(music_signal_n), Fs_orig); % Fs_orig is the original sampling frequency

% % Obtain Voice Signal
% voice_signal = y - music_signal; % Subtract music signal from the original mixture signal
% 
% %  rest of the reconstruction and audio file writing code
% voice_signal_n = voice_signal / max(abs(voice_signal)); % Normalize the signal
% 
% % Save Voice Signal
% voice_filepath = "voice_signal.wav"; % Choose a file name and path
% audiowrite(voice_filepath, real(voice_signal_n), Fs_orig); % Fs_orig is the original sampling frequency



