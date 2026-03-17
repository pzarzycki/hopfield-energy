import { readFile } from "node:fs/promises";

import { getDatasetAssetPath, parseDatasetArchiveBuffer, type DatasetArchive, type DatasetName } from "../../src/data/datasetArchives";

export async function loadDatasetArchiveFromNodeFs(dataset: DatasetName): Promise<DatasetArchive> {
  const path = new URL(`../../public/${getDatasetAssetPath(dataset)}`, import.meta.url);
  const bytes = await readFile(path);
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
  return parseDatasetArchiveBuffer(buffer);
}
