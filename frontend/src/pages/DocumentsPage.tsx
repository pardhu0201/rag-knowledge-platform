import { useCallback, useEffect, useRef, useState } from "react";
import { FileText, Loader2, Trash2, UploadCloud } from "lucide-react";

import { EmptyState, PageHeader } from "../components/primitives";
import { api, type DocumentSummary } from "../lib/api";

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [status, setStatus] = useState("");
  const [uploading, setUploading] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setDocuments(await api.documents());
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const upload = async (file: File) => {
    setUploading(true);
    setStatus("");
    try {
      const result = await api.uploadDocument(file);
      setStatus(`${result.status}: "${result.title}" indexed into ${result.chunks} chunks.`);
      await load();
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  };

  const remove = async (id: string, title: string) => {
    if (!window.confirm(`Remove "${title}" and all its indexed chunks?`)) return;
    await api.deleteDocument(id);
    await load();
  };

  const totalChunks = documents.reduce((sum, d) => sum + d.chunk_count, 0);
  const totalPages = documents.reduce((sum, d) => sum + d.page_count, 0);

  return (
    <div className="min-h-screen">
      <PageHeader
        title="Documents"
        subtitle={`${documents.length} documents, ${totalPages} pages, ${totalChunks} indexed chunks. Upload a PDF, DOCX or text file to make it answerable immediately.`}
        actions={
          <>
            <input
              ref={fileInput}
              type="file"
              accept=".pdf,.docx,.txt,.md"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) void upload(file);
              }}
            />
            <button
              className="btn-primary"
              onClick={() => fileInput.current?.click()}
              disabled={uploading}
            >
              {uploading ? (
                <Loader2 size={14} className="animate-spin" />
              ) : (
                <UploadCloud size={14} />
              )}
              Upload document
            </button>
          </>
        }
      />

      <div className="px-5 py-6 sm:px-7">
        {status && (
          <p className="mb-4 rounded-lg border border-[var(--color-line-strong)] bg-[var(--color-surface-2)] px-3.5 py-2.5 text-[13px] text-[var(--color-ink-2)]">
            {status}
          </p>
        )}

        {loading ? (
          <div className="flex justify-center py-12 text-[var(--color-ink-3)]">
            <Loader2 size={20} className="animate-spin" />
          </div>
        ) : documents.length === 0 ? (
          <EmptyState
            icon={<FileText size={18} />}
            title="No documents indexed"
            body="Upload a PDF, DOCX or text file to populate the knowledge base."
          />
        ) : (
          <ul className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
            {documents.map((doc) => (
              <li key={doc.id} className="panel-2 flex items-start gap-3 p-3.5">
                <span className="mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-[var(--color-surface-3)] text-[var(--color-ink-3)]">
                  <FileText size={14} />
                </span>
                <div className="min-w-0 flex-1">
                  <div className="truncate text-[13.5px] font-medium">{doc.title}</div>
                  <div className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-[var(--color-ink-3)]">
                    <span className="uppercase">{doc.doc_type}</span>
                    <span>&middot;</span>
                    <span>{doc.page_count} pages</span>
                    <span>&middot;</span>
                    <span className="font-mono">{doc.chunk_count} chunks</span>
                  </div>
                </div>
                <button
                  onClick={() => void remove(doc.id, doc.title)}
                  className="shrink-0 rounded-md p-1.5 text-[var(--color-ink-3)] transition-colors hover:bg-[rgba(225,29,72,0.12)] hover:text-[var(--color-danger)]"
                  aria-label={`Remove ${doc.title}`}
                >
                  <Trash2 size={14} />
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
