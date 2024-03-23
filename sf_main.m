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
disp('Size of final beat spectrum is');
bs_size = size(bs_r_n);
disp(bs_size);

% Plot the final beat spectrum
t = 1:bs_size;
figure;
plot(t, bs_r_n);
xlabel('Index');
ylabel('Value');
title('Final Beat Spectrum');
hold on;

% Find peaks in the beat spectrum
bs_real = real(bs_r_n);
[peaks, locs] = findpeaks(bs_real);
disp("No. of local peaks found is");
disp(size(peaks));

% Find which specific peaks are occuring periodically
% Method not specified in paper
% Making an assumption that all clips will be above 5s
dur = length(y)/Fs; % total duration of audio signal in s
s_per_five = round(bs_size(1)/dur*5);  % No. of rows in 5s
size_ref = bs_size(1);
repeating_peaks = zeros(bs_size(1)+1);
repeating_locs = zeros(bs_size(1)+1);
loc_repeated_at = zeros(bs_size(1)+1);
allocated_till = 1;
for i = 1:size(peaks)
    peak = peaks(i);
    loc = locs(i);
    limit = loc + s_per_five;
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
repeating_period = loc_repeated_at(k)-repeating_locs(k);
% periods = diff(locs);

% Determine the repeating period
bs_per_five = round(bs_size(1)/dur);  % No. of rows in 1s

% Use the repeating period to evenly time-segment the mixture spectrogram
% num_segments = floor(size(S, 2) / repeating_period);
% segment_length = repeating_period;

% Initialize segmented spectrogram matrix
% segmented_S = zeros(size(S, 1), num_segments, segment_length);

% Segment the mixture spectrogram
% for i = 1:num_segments
%     start_index = (i - 1) * repeating_period + 1;
%     end_index = start_index + segment_length - 1;
%     segmented_S(:, i, :) = S(:, start_index:end_index);
% end

% Process each segment as needed

% Continue with further processing or analysis





