# print(snakemake.input[0])
# print(snakemake.output[0])
# print(snakemake.dataset)

print("hey")
print(snakemake.output)
f=open(snakemake.output[0],'w')
f.write("aaasddfs\n")
f.close()