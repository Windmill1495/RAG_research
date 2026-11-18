"""Identity and counts of the audited official QReCC archive."""

ARCHIVE_MD5 = "16c48fb9c4b9603b8ccc8ba3bd7d69c1"
ARCHIVE_SIZE = 27_512_656_777
VALID_ROWS = 54_241_590
UNIQUE_VARIANTS = 54_226_214
VARIANT_POLICY = "official-id-and-exact-text-v1"


def is_official_archive(identity):
    files = identity.get("files", [])
    return (len(files) == 1 and isinstance(files[0], dict) and files[0].get("md5") == ARCHIVE_MD5
            and files[0].get("bytes") == ARCHIVE_SIZE)
