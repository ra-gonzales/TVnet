% Enable GPU processing
GPU = true;

% Load TVnet networks
load(fullfile(pwd, 'models', 'TVnet_4ch_1st.mat'));
load(fullfile(pwd, 'models', 'TVnet_4ch_2nd.mat'));

% Load anonymized dicom data from a subject
data_sample_path = fullfile(pwd, '4ch_data_sample');
[IM_4ch, Rxy_4ch, TimeVector_4ch] = TVnet_functions('load_dicom_data', data_sample_path);

% Track the tricuspid valve
TV_4ch = TVnet_functions('pipeline',IM_4ch,Rxy_4ch,TVnet_4ch_1st,TVnet_4ch_2nd,GPU);

% Visualize the tracking
figure,
for i=1:size(TV_4ch,1)
    imagesc(IM_4ch(:,:,i)),colormap(gray), hold on,
    plot(TV_4ch(i,2),TV_4ch(i,1),'*r'), hold on,
    plot(TV_4ch(i,4),TV_4ch(i,3),'*g'),
    pause(0.2);
end
