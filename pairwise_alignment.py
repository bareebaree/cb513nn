import random
import datetime
import multiprocessing
from Bio import SeqIO, pairwise2
from tqdm import tqdm  # For progress bars
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

"""
This script uses the pairwise alignment module from Biopython to calculate the best match for a query sequence against a set database,
and calculates this by by using a scoring module in Biopython. It then calculates the P value of the alignment appearing by chance.

"""


def shuffle_sequence(seq) -> str:
    """Shuffle a sequence while preserving its composition.
    
    PARAMETERS: seq ->(str): Any given sequence from the ID:sequence dictionary
    
    RETURNS:  Shuffled sequence appended to a shuffled sequence list.
    
    """
    seq_list = list(seq)
    random.shuffle(seq_list)
    return "".join(seq_list)

def shuffle_and_align():
    """Shuffles the query sequence and computes an alignment score. Calls shuffle_sequence() to do so.
       Scoring scheme = match = +1
       gap/mismatch = 0
       
    PARAMETERS: query_seq -> (str): The query sequence.
    
    RETURNS: The score of each shuffled query 
    
    """
    shuffled_query = shuffle_sequence(query_seq)
    return pairwise2.align.globalxx(shuffled_query, best_match_seq, score_only=True)


# File paths
database_path= "./data/dog_breeds.fa"
query_path = "./data/mystery.fa"


# Read the FASTA database into a dictionary of key:value pairs for ID:sequence, and iterativly convert each record into a string.
database = {record.id: str(record.seq) for record in SeqIO.parse(database_path, "fasta")}

# Read the query sequence (assuming only one sequence in the file), and converts it into a string.
query_record = next(SeqIO.parse(query_path, "fasta"))  
query_seq = str(query_record.seq)  

# Initialise the values for best match and the best score.
best_match = None
best_score = float('-inf')

print("Searching for the best match in the database...")

# Find the most similar sequence (with progress bar)
for name, seq in tqdm(database.items(), desc="Aligning sequences", unit="seq"):
    score = pairwise2.align.globalxx(query_seq, seq, score_only=True)
    
    if score > best_score:
        best_score = score
        best_match = name

# Retrieve the best matching sequence
best_match_seq = database[best_match]

print(f"\nBest match: {best_match} with score: {best_score}")

# Perform full sequence alignment
alignments = pairwise2.align.globalxx(query_seq, best_match_seq)

# Get the best alignment, chooses first value in alignments
best_alignment = alignments[0]
aligned_query_seq = best_alignment.seqA
aligned_best_match_seq = best_alignment.seqB

# Print the best alignment
print("\nBest alignment:")
print(pairwise2.format_alignment(*best_alignment))

# Compute percent identity by calculating the % of matches against sequence length
num_matches = sum(1 for a, b in zip(aligned_query_seq, aligned_best_match_seq) if a == b)
total_length = max(len(aligned_query_seq), len(aligned_best_match_seq))
percent_identity = (num_matches / total_length) * 100

print(f"\nPercent Identity: {percent_identity:.2f}%")

# ---------------- Empirical P-value Calculation ----------------

# This section calculates P-value by shuffling the sequences and matching the query sequence against them. The P-value
# is calculated by the number of random sequences that had a greater score against the database than the query sequence.


# Number of shuffles allocated.

num_shuffles = 1

print("\nComputing empirical P-value with shuffled sequences...")

# Sequential calculation with progress bar, loops calling shuffle_and_align()
random_scores = []
for _ in tqdm(range(num_shuffles), desc="Shuffling and aligning", unit="shuffle"):
    score = shuffle_and_align()
    random_scores.append(score)

# Compute empirical P-value
empirical_p_value = sum(s >= best_score for s in random_scores) / num_shuffles

print(f"\nEmpirical P-value: {empirical_p_value:.8f}")

# ---------------- Save results ----------------

# Define a timestamp to create an identifier for job run.
current_datetime = datetime.datetime.now()
timestamp = current_datetime.strftime("%Y%m%d_%H%M%S")

# Create SeqRecords for the aligned query and the best-match sequence
query_record_aligned = SeqRecord(Seq(aligned_query_seq),
                                 id=f"{query_record.id}_aligned",
                                 description="Aligned query sequence")

best_match_record_aligned = SeqRecord(Seq(aligned_best_match_seq),
                                      id=f"{best_match}_aligned",
                                      description="Aligned best match sequence")

# Generate filename for the alignment FASTA file
alignment_fasta_file = f"./results/{timestamp}_alignment.fasta"

# Write aligned query and aligned best-match sequences to FASTA
with open(alignment_fasta_file, "w") as fasta_out:
    SeqIO.write([query_record_aligned, best_match_record_aligned], fasta_out, "fasta")

print(f"\nAlignment saved as FASTA file: {alignment_fasta_file}")

# Save results summary in a text file
output_file = f"./results/{timestamp}_alignment_results.txt"

with open(output_file, "w") as f:
    f.write(f"Query ID: {query_record.id}\n")
    f.write(f"Query sequence: {query_seq}\n")
    f.write(f"Best match: {best_match}\n")
    f.write(f"Alignment Score: {best_score}\n")
    f.write(f"Percent Identity: {percent_identity:.2f}%\n")
    f.write(f"Empirical P-value: {empirical_p_value:.8f}\n\n")
    f.write("Best Alignment:\n")
    f.write(pairwise2.format_alignment(*best_alignment))

print(f"\nResults summary saved to {output_file}")


