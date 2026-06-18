import pathlib
p = pathlib.Path("src/astra/app/uploads.py")
text = p.read_text()
text = text.replace(
    "tmp = self._meta_path(record.upload_id).with_suffix(\".meta.json.tmp\")",
    "tmp = self._meta_path(record.upload_id).with_name(self._meta_path(record.upload_id).name + \".tmp\")",
)
p.write_text(text)
print("OK")
