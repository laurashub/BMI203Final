
def read_seqs(filename):
	"""
	A few assumptions here 
	"""
	seqs = []

	#if its a fasta, be careful
	if filename[:-2] == 'fa':
		with open(filename) as f:
			seq = ""
			lines = f.read().splitlines()
			for line in lines:
				if '>' in line and seq != "":
					seqs.append(seq)
					seq = ""
				else:
					seq += line
			seqs.append(seq)
	else:
		seqs = open(filename).read().splitlines()
	return seqs