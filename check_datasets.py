import mirdata

try:
    mdb = mirdata.initialize('mdb_drums')
    print("MDB-Drums info:", mdb.readme)
    # Check if download is possible
    # mdb.download()  # Only dry run or check if it requires manual steps
    
    idmt = mirdata.initialize('idmt_smt_drums')
    print("IDMT-SMT-Drums info:", idmt.readme)
except Exception as e:
    print("Error:", e)
