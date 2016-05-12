data = open("data.fasta")
dataLines = data.readlines()
ss = open("ss.txt")
ssLines = ss.readlines()
train = open("train.fasta","w")
ssTrain = open("ssTrain.txt","w")

for k,s in enumerate(ssLines):
	seq = ""
	ssSeq = ""
	if ">" not in s or "sequence" not in s:
		continue
	j = k
	j+=1
	while ">" not in ssLines[j]:
		seq += ssLines[j].strip()
		j+=1
	j+=1
	while j < len(ssLines) and ">" not in ssLines[j] :
		ssSeq += ssLines[j].strip()
		j+=1
	s = s.split(":")

	for i,d in enumerate(dataLines):
		if ">" not in d:
			continue
		d = d.split()
		if(s[0]+s[1] == d[0]):
			print("FIND")
			train.write(dataLines[i])
			train.write(seq+"\n\n")
			ssTrain.write(dataLines[i])
			ssTrain.write(ssSeq+"\n\n")
	if(k%1024==0):
		print(k)