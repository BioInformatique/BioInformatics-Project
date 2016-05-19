from Bio.Blast.Applications import NcbipsiblastCommandline
from Bio import SeqIO
from Bio.Blast import NCBIXML
import numpy as np
from scipy.sparse import csr_matrix

AA = "ABCDEFGHIJKLMNOPQRSTUVWYZX*-"
AA_MAPPING = dict((s,i) for i,s in enumerate(AA))

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def computeProfile(record, querySeq ,weights = None):
    profile = np.zeros(shape = (28,record.query_letters))
    if weights == None:
        weights = [1]*(len(record.alignments)+1)

    for k,al in enumerate(record.alignments):
        for hsp in al.hsps:
            i = hsp.query_start-1
            for aaQ, aaH in zip(hsp.query,hsp.sbjct):
                if(aaQ=='-'):
                    continue
                profile[AA_MAPPING[aaH]][i] += 1*weights[k]
                i+=1

    for i,aa in enumerate(querySeq):
        profile[AA_MAPPING[aa]][i]+=1*weights[-1]

    profile = profile/profile.sum(axis=0)
    return profile

def computeWeights(record, querySeq, profile):
    profile = np.log(profile)
    weights = [0]*(len(record.alignments)+1)

    weight = 0
    for i,aa in enumerate(querySeq):
        weight+=profile[AA_MAPPING[aa]][i]
    weights[-1] = -weight

    for k,al in enumerate(record.alignments):
        weight = 0
        for hsp in al.hsps:
            i = hsp.query_start-1
            for aaQ, aaH in zip(hsp.query,hsp.sbjct):
                if(aaQ=='-'):
                    continue
                if(aaH in AA):
                    weight+=profile[AA_MAPPING[aaH]][i]
                i+=1
        weights[k] = -weight
    return weights

def psiBlast(filename, querySeq):
    global AA, AA_MAPPING
    pssmFileName = "temp/pssm"
    cline = NcbipsiblastCommandline('psiblast',db="dbCoil/nr", query = "temp/query.fasta", \
            out = "temp/out.xml", out_pssm = pssmFileName, num_iterations = 3, seg = "yes", \
            evalue = 10**-3, outfmt = "5")
    cline()
    cline = NcbipsiblastCommandline('psiblast',db="db/nr", \
            out = "temp/outFinal.xml", in_pssm = pssmFileName,out_pssm = pssmFileName+"Final", num_iterations = 1, outfmt = "5")
    cline()


    res = open("temp/outFinal.xml", "r")
    record = NCBIXML.read(res)
    profile = computeProfile(record,querySeq )
    weights = computeWeights(record, querySeq, profile)
    profile = computeProfile(record, querySeq, weights)
    profile = csr_matrix(profile)
    save_sparse_csr(filename,profile)

def main():
    seqs = list(SeqIO.parse("data/train50.fasta", "fasta"))
    for i,seq in enumerate(seqs):
        tempFile = open("temp/query.fasta","w")
        tempFile.write(">"+seq.id+"\n"+str(seq.seq)+"\n")
        tempFile.close()
        psiBlast("profile/"+seq.id,seq.seq)
        print(i,end=" - ")
        print(seq.id)



if __name__ == '__main__':
    main()

# psiBlast("test/Profile")
# profileLoad = load_sparse_csr("test/Profile.npz")