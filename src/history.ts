export type HistoryRecord = {
  id: string;
  name: string;
  createdAt: number;
  duration: number;
  bytes: number;
  vocals: Blob;
  instrumental: Blob;
};

export type HistorySummary = Omit<HistoryRecord, "vocals" | "instrumental">;

const DB_NAME = "subtract-history";
const DB_VERSION = 1;
const STORE = "results";
let databasePromise: Promise<IDBDatabase> | undefined;

const requestResult = <T>(request: IDBRequest<T>): Promise<T> => new Promise((resolve, reject) => {
  request.addEventListener("success", () => resolve(request.result), { once: true });
  request.addEventListener("error", () => reject(request.error), { once: true });
});

const transactionDone = (transaction: IDBTransaction): Promise<void> => new Promise((resolve, reject) => {
  transaction.addEventListener("complete", () => resolve(), { once: true });
  transaction.addEventListener("abort", () => reject(transaction.error), { once: true });
  transaction.addEventListener("error", () => reject(transaction.error), { once: true });
});

const openDatabase = (): Promise<IDBDatabase> => {
  if (databasePromise) return databasePromise;
  databasePromise = new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.addEventListener("upgradeneeded", () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(STORE)) {
        const store = database.createObjectStore(STORE, { keyPath: "id" });
        store.createIndex("createdAt", "createdAt");
      }
    });
    request.addEventListener("success", () => resolve(request.result), { once: true });
    request.addEventListener("error", () => reject(request.error), { once: true });
  });
  return databasePromise;
};

export async function saveHistoryRecord(record: HistoryRecord): Promise<void> {
  const database = await openDatabase();
  const transaction = database.transaction(STORE, "readwrite");
  transaction.objectStore(STORE).put(record);
  await transactionDone(transaction);
}

export async function getHistoryRecord(id: string): Promise<HistoryRecord | undefined> {
  const database = await openDatabase();
  const transaction = database.transaction(STORE, "readonly");
  return requestResult(transaction.objectStore(STORE).get(id)) as Promise<HistoryRecord | undefined>;
}

export async function listHistoryRecords(): Promise<HistorySummary[]> {
  const database = await openDatabase();
  const transaction = database.transaction(STORE, "readonly");
  const records = await requestResult(transaction.objectStore(STORE).getAll()) as HistoryRecord[];
  return records
    .map(({ id, name, createdAt, duration, bytes }) => ({ id, name, createdAt, duration, bytes }))
    .sort((a, b) => b.createdAt - a.createdAt);
}

export async function deleteHistoryRecord(id: string): Promise<void> {
  const database = await openDatabase();
  const transaction = database.transaction(STORE, "readwrite");
  transaction.objectStore(STORE).delete(id);
  await transactionDone(transaction);
}

export async function clearHistoryRecords(): Promise<void> {
  const database = await openDatabase();
  const transaction = database.transaction(STORE, "readwrite");
  transaction.objectStore(STORE).clear();
  await transactionDone(transaction);
}
