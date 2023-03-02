close all; clear all; clc;

%load('test_apce_enc200ep_nonorm_noupdate.mat');
load('test_DCFnet200_noupdate.mat');

fn = fieldnames(PSR_p_lists);
avgFP = 0;
avgFN = 0;
avgLE = 0;
avg_enc_acc = 0;
avg_apce_acc = 0;
avg_psr_acc = 0;
avg_APCE_n = 0;
avg_APCE_p = 0;
total_p_entries = 0;
total_n_entries = 0;
FPs = zeros(numel(fn),1);
FNs = zeros(numel(fn),1);
thresholds = zeros(numel(fn),1);

usePSR = false;


for k = 1:numel(fn)
    PSR_p_list = PSR_p_lists.(fn{k});
    PSR_n_list = PSR_n_lists.(fn{k});
    APCE_p_list = APCE_p_lists.(fn{k});
    APCE_n_list = APCE_n_lists.(fn{k});
    pos_diff_p_list = pos_diff_p_lists.(fn{k});
    accs = acc_list.(fn{k});
    enc_dist_acc = accs(1);
    apce_acc = accs(2);
    psr_acc = accs(3);
    avg_enc_acc = avg_enc_acc + enc_dist_acc*length(PSR_p_list);
    avg_apce_acc = avg_apce_acc + apce_acc*length(PSR_p_list);
    avg_psr_acc = avg_psr_acc + psr_acc*length(PSR_p_list);
%     figure()
%     histogram(PSR_p_list, 'BinWidth',0.05);hold on;
%     histogram(PSR_n_list, 'BinWidth',0.05);
%     legend('p','n');
    %figure()
    %histogram(APCE_p_list, 'BinWidth',0.5);hold on;
    %histogram(APCE_n_list, 'BinWidth',0.5);
    %legend('p','n');
    max_gm = 0;
    if usePSR
        th_range = min([PSR_n_list,PSR_p_list]):0.05:max([PSR_n_list,PSR_p_list]);
    else
        th_range = min([APCE_n_list,APCE_p_list]):0.05:max([APCE_n_list,APCE_p_list]);
    end
    best_thresh = 0;
    best_fnr = 1;
    best_fpr = 1;
    FNrs = zeros(length(th_range),1);
    FPrs = zeros(length(th_range),1);
    for i = 1:length(th_range)
        thresh = th_range(i);
        if usePSR
            FNr = length(PSR_p_list(PSR_p_list<=thresh))/length(PSR_p_list);
            FPr = length(PSR_n_list(PSR_n_list>thresh))/length(PSR_n_list);
        else
            FNr = length(APCE_p_list(APCE_p_list<=thresh))/length(APCE_p_list);
            FPr = length(APCE_n_list(APCE_n_list>thresh))/length(APCE_n_list);
        end
        TPr = 1-FNr;
        TNr = 1-FPr;
        FNrs(i) = FNr;
        FPrs(i) = FPr;
        gm = sqrt(TPr*TNr);
        if gm > max_gm
            max_gm=gm;
            best_thresh = thresh;
            best_fnr = FNr;
            best_fpr = FPr;
        end
    end
%     figure()
%     plot(th_range, FNrs,'r'); hold on;
%     plot(th_range, FPrs,'b');
    display(fn{k})
    display(best_fnr)
    display(best_fpr)
    FNs(k) = best_fnr;
    FPs(k) = best_fpr;
    thresholds(k) = best_thresh;
    avg_APCE_n = avg_APCE_n + sum(APCE_n_list);
    avg_APCE_p = avg_APCE_p + sum(APCE_p_list);
    avgLE = avgLE + sum(pos_diff_p_list);
    avgFP = avgFP + length(PSR_n_list)*best_fpr;
    avgFN = avgFN + length(PSR_p_list)*best_fnr;
    total_p_entries = total_p_entries + length(PSR_p_list);
    total_n_entries = total_n_entries + length(PSR_n_list);            
end
avgFP = avgFP/total_n_entries
avgFN = avgFN/total_p_entries
avgLE = avgLE/total_p_entries
avgGM = sqrt((1-avgFP)*(1-avgFN))
avg_enc_acc = avg_enc_acc/total_p_entries
avg_apce_acc = avg_apce_acc/total_p_entries
avg_psr_acc = avg_psr_acc/total_p_entries
avg_APCE_p = avg_APCE_p/total_p_entries
avg_APCE_n = avg_APCE_n/total_n_entries

%%
close all; clear all; clc;
load('test_FathomNet_Hog_update.mat');

x_values =  {'bathochordaeus filter', 'prayidae',...
    'poeobius', 'mitrocoma', 'cydippida', 'calycophorae', 'beroe', ...
    'bathochordaeus', 'atolla', 'aegina', 'solmissus', ...
    'lampocteis cruentiventer'};
y_values = {'bathochordaeus filter', 'prayidae', ...
    'poeobius', 'mitrocoma', 'cydippida', 'calycophorae', 'beroe', ...
    'bathochordaeus', 'atolla', 'aegina', 'solmissus', ...
    'lampocteis cruentiventer'};

%psr_map = heatmap(x_values, y_values, PSR_conf);
apce_map = heatmap(x_values,y_values, APCE_conf)
%%
close all; clear all; clc;
load('test_Rotation.mat');
angle = 1:1:5;
SCF_APCE_mean = mean(SCF_APCE,1);
SCF_APCE_std = std(SCF_APCE,1);
% xconf = [angle angle(end:-1:1)] ;         
% yconf = [SCF_APCE_mean+SCF_APCE_std SCF_APCE_mean(end:-1:1)-SCF_APCE_std];

figure()
% p = fill(xconf,yconf,'g'); hold on;
% p.FaceColor = [1 0.8 0.8];      
% p.EdgeColor = 'none';
h1 = plot(angle, median(DCF_APCE,1),'Color',[0.8500 0.3250 0.0980]); hold on;
plot(angle, ones(5,1)*8.2123, '--', 'Color', [0.8500 0.3250 0.0980]);
h2 = plot(angle, median(hog_APCE,1),'Color',[0 0.4470 0.7410]);
plot(angle, ones(5,1)*11.2728,'--', 'Color',[0 0.4470 0.7410]);
h3 = plot(angle, median(SCF_APCE,1),'Color','k');
plot(angle, ones(5,1)*9.6672,'--', 'Color','k');
% for j = 1:length(DCF_APCE(:,1))
%     plot(angle, DCF_APCE(j,:),'r.');
%     plot(angle, hog_APCE(j,:),'b.');
%     plot(angle, SCF_APCE(j,:),'g.');
% end
legend('DCFNet','DCFNet threshold','HoG','HoG threshold','SqueezeCFNet','SqueezeCFNet threshold');
names = {'90^\circ CW'; '180^\circ'; '90^\circ CCW'; 'VF'; 'HF'};
set(gca,'xtick',[1:5],'xticklabel',names)
ylabel('APCE')
% hog APCE thresh avg = 11.2728, avg P=23.4, avg N=9
% SCF thresh avg = 9.6672, avg P = 11.9, avg N = 8.7
% DCF thresh avg = 8.2123, avg P = 24.2, avg N = 7.4

%%
DCF_all = DCF_APCE(:);
DCF_thresh = 8.2123;
FN_DCF = length(DCF_all(DCF_all<DCF_thresh))/length(DCF_APCE(:))

hog_thresh = 11.2728;
FN_HOG = length(hog_APCE(hog_APCE<hog_thresh))/length(hog_APCE(:))

SCF_thresh =  9.6672;
FN_SCF = length(SCF_APCE(SCF_APCE<SCF_thresh))/length(SCF_APCE(:))

std_DCF = std(DCF_APCE,[],'all')
std_HOG = std(hog_APCE,[],'all')
std_SCF = std(SCF_APCE,[],'all')

mean_DCF = mean(DCF_APCE,'all')
mean_HOG = mean(hog_APCE,'all')
mean_SCF = mean(SCF_APCE,'all')
%%
% figure()
% plot(angle, mean(DCF_PSR,1),'r'); hold on;
% plot(angle, mean(hog_PSR,1),'b');
% plot(angle, mean(SCF_PSR,1),'g');
% legend('DCFNet','HoG','SqueezeCFNet');
%%
figure()
histogram(DCF_APCE(:),'BinWidth',0.5); hold on;
histogram(hog_APCE(:),'BinWidth',0.5);
histogram(SCF_APCE(:),'BinWidth',0.5);

