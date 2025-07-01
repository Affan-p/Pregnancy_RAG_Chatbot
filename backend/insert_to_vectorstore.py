from backend.utils import rebuild_index

# This will:
# 1. Delete all old vectors from Pinecone
# 2. Clear local storage
# 3. Process your CSVs with the new CSV reader
# 4. Rebuild and store everything fresh
index = rebuild_index()

if index:
    print("🎉 Ready to go! Your CSVs are now properly processed")
else:
    print("❌ Something went wrong with the rebuild")