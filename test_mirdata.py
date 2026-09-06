import mirdata

print("Available datasets:")
print([d for d in mirdata.list_datasets() if 'drum' in d])
