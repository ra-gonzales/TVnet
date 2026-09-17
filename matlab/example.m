% Track the tricuspid valve in the bundled four-chamber cine and show the result.

here = fileparts(mfilename('fullpath'));
addpath(here);
use_gpu = true;

% The two trained networks.
load(fullfile(here, 'models', 'TVnet_4ch_1st.mat'), 'TVnet_4ch_1st');
load(fullfile(here, 'models', 'TVnet_4ch_2nd.mat'), 'TVnet_4ch_2nd');

% An anonymized four-chamber cine, one cardiac cycle of single-frame DICOMs.
[IM, Rxy, time_vector] = TVnet_functions('load_dicom_data', ...
                                         fullfile(fileparts(here), '4ch_data_sample'));

% TV is (frames, 4) = [row_septal, col_septal, row_lateral, col_lateral].
TV = TVnet_functions('pipeline', IM, Rxy, TVnet_4ch_1st, TVnet_4ch_2nd, use_gpu);

figure;
for i = 1:size(TV,1)
    imagesc(IM(:,:,i)); colormap gray; axis image off; hold on;
    plot(TV(i,2), TV(i,1), '*', 'Color', [0.20 0.70 0.30]);   % septal, as in Fig. 1
    plot(TV(i,4), TV(i,3), '*', 'Color', [0.30 0.70 0.90]);   % lateral, as in Fig. 1
    hold off;
    title(sprintf('frame %d of %d   t = %.0f ms', i, size(TV,1), time_vector(i)*1000));
    pause(0.2);
end
