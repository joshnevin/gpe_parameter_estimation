function P=DJI_OSA2(Signal, P)

%file: DJI_OSA2.m
%Version: 4.0
%Date: 17 Apr 2015
%Author: David Ives
%email: d.ives@ee.ucl.ac.uk

%v4 units to SI
%v2 error factor of 2 in Resolution bandwidth corrected, dont know how it
% got there as DJI_ESA version is OK.

%Purpose: To display a plot of optical power (dBm) vs frequency (GHz).
% to continuously average.

% Input:	Signal.dT = sample spacing (s)
%			Signal.Et(n,:) = fields (sqrt(W))

% Parameters 	P.RB = Resolution Bandwidth (Hz).
%				P.Na = Number of previous averages (Optional)
%				P.P  = Optical Power (W) from previous averages (Optional)

% Output: A graph, Optical Power (P.P = W); Sample Spacing (P.dF = Hz);
%   Actual Resolution BW (P.RBa = Hz); Number of averages (P.Na)

% take smaller windows to set RB and average along time domain trace.
% window width 2400/(RB.dT)
% window is cos^2*gaussian to ensure drop to zero at edges and give low sidebands
% sidelobes at -100dB
% only 1/3 data in window is used so overlap windows by 2/3 to get full averageing
%

N=length(Signal.Et);

if (nargin<2)
	P.Na=0;
end

if ~isfield(P, 'Na')
	P.Na=0;
end

if ~isfield(P, 'RB')
	P.RB=(2.3985/(N*Signal.dT));
end

if P.RB<(2.3985/(N*Signal.dT))
    warning('Resolution Bandwidth too small')
    P.RB=(2.3985/(N*Signal.dT));
end

%Generate gaussian * Hann window function
WN=3*round(0.800/(P.RB*Signal.dT));   % window size in points
NN=floor(3*(N-WN)/WN);  % number of windows within data span
SN=round((N-NN*WN/3-WN)/2+1); % offset to first window to use central part of data

G=(cos((1-WN/2:WN/2)*pi/WN).*cos((1-WN/2:WN/2)*pi/WN)).*exp(-(1-WN/2:WN/2).*(1-WN/2:WN/2)*(25/(WN*WN))); 
%G=(cos((1-WN/2:WN/2)*pi/WN).*cos((1-WN/2:WN/2)*pi/WN)); 
G=ones(size(Signal.Et,1),1)*(WN/sum(G))*G;

if (~isfield(P, 'P') || P.Na<1)
	P.P=zeros(size(Signal.Et,1),WN*2);
end

Pl=zeros(size(Signal.Et,1),WN*2);

for i=1:NN;
	% Apply Window
	VG=Signal.Et(:,(SN+(i-1)*WN/3):(SN+(i-1)*WN/3+WN-1)).*G;

	% Calulate spectrum 2*oversampled
	A=fftshift(fft(VG,WN*2,2)./WN);

	% running average with input optical power P
	Pl=real(A(:,1:WN*2).*conj(A(:,1:WN*2)))+Pl;

end

P.P=(1/(P.Na+1))*(1/NN)*Pl+(P.Na/(P.Na+1))*P.P;
P.Na=P.Na+1;

P.PdBm=10*log10(sum(P.P))+30; % convert to dBm for graph
P.F=(-WN*1:WN*1-1)/(WN*2*Signal.dT); % Signal.dT is in s, factor 2 for oversampling in fft to get to Hz
    
%Calculate actual Resolution Bandwidth
FG=fft(G(1,:));
P.RBa=sum(FG.*conj(FG))/(WN*Signal.dT*FG(1)*conj(FG(1))); % Hz

% %Plot Graph
% plot(P.F,P.PdBm);
% grid on
% xlabel('Frequency (Hz)');
% ylabel('Power (dBm)');
% title(sprintf('Virtual Optical Spectum Analyser, RB = %0.4g (Hz), Navg = %g',P.RBa,P.Na));

end



