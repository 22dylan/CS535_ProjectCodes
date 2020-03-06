
storms = readmatrix('.\data\NACCS_TS_Sim0_Post0_ST_TROP_STcond.csv');
datapoints = readmatrix('.\data\NACCS_SavePts_18977_ConversionKey_MSL_to_NAVD_meters.csv');

latlim = [-78 -66];
lonlim = [34 46];

figure
hold on
% usamap(latlim, lonlim)
geoshow('landareas.shp', 'FaceColor', [.75 1.0 0.5]);
ylim(lonlim)
xlim(latlim)
geoshow(datapoints(:,2),datapoints(:,3),'Marker','.','MarkerSize',2,'LineStyle','none')
for i=1:50:500
    storm = storms(storms(:,3)==i,:);
    geoshow(storm(:,15),storm(:,16))
end
hold off
% 
% figure
% plot(surge(:,1))
