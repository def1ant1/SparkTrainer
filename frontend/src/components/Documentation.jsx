import React, { useState, useEffect } from 'react';
import { BookOpen, ChevronRight, ChevronDown, Home, Search } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

/**
 * Documentation Viewer Component
 *
 * Displays in-app documentation with:
 * - Sidebar navigation with sections and subsections
 * - Markdown rendering with GitHub Flavored Markdown support
 * - Search functionality
 * - Deep linking to specific sections
 */
export default function Documentation({ onNavigate }) {
  const [docStructure, setDocStructure] = useState(null);
  const [currentDoc, setCurrentDoc] = useState(null);
  const [currentSection, setCurrentSection] = useState(null);
  const [docContent, setDocContent] = useState('');
  const [expandedSections, setExpandedSections] = useState({});
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(false);

  // Load documentation structure on mount
  useEffect(() => {
    loadDocStructure();
  }, []);

  // Load initial doc (overview) when structure is loaded
  useEffect(() => {
    if (docStructure && !currentDoc) {
      loadDoc('overview', docStructure.sections[0]);
    }
  }, [docStructure]);

  async function loadDocStructure() {
    try {
      // In production, this would be served statically
      const response = await fetch('/docs/app/index.json');
      const structure = await response.json();
      setDocStructure(structure);

      // Expand first section by default
      const initialExpanded = {};
      structure.sections.forEach(s => {
        if (s.subsections) initialExpanded[s.id] = true;
      });
      setExpandedSections(initialExpanded);
    } catch (error) {
      console.error('Failed to load documentation structure:', error);
    }
  }

  async function loadDoc(docId, section, subsection = null) {
    setLoading(true);
    try {
      const file = subsection ? subsection.file : section.file;
      const response = await fetch(`/docs/app/${file}`);
      const content = await response.text();
      setDocContent(content);
      setCurrentDoc(docId);
      setCurrentSection({ ...section, subsection });
    } catch (error) {
      console.error('Failed to load documentation:', error);
      setDocContent('# Error\n\nFailed to load documentation. Please try again.');
    } finally {
      setLoading(false);
    }
  }

  function toggleSection(sectionId) {
    setExpandedSections(prev => ({
      ...prev,
      [sectionId]: !prev[sectionId]
    }));
  }

  function renderSidebarItem(section, level = 0) {
    const hasSubsections = section.subsections && section.subsections.length > 0;
    const isExpanded = expandedSections[section.id];
    const isActive = currentSection?.id === section.id;

    return (
      <div key={section.id}>
        <div
          className={`flex items-center gap-2 px-3 py-2 cursor-pointer rounded-lg transition-colors ${
            isActive ? 'bg-primary/10 text-primary' : 'hover:bg-muted text-text/70 hover:text-text'
          }`}
          style={{ paddingLeft: `${level * 1.5 + 0.75}rem` }}
          onClick={() => {
            if (hasSubsections) {
              toggleSection(section.id);
            } else {
              loadDoc(section.id, section);
            }
          }}
        >
          {hasSubsections && (
            <span className="text-text/50">
              {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
            </span>
          )}
          <span className="text-sm flex-1">{section.title}</span>
        </div>

        {hasSubsections && isExpanded && (
          <div className="mt-1 space-y-1">
            {section.subsections.map(sub => (
              <div
                key={sub.id}
                className={`flex items-center gap-2 px-3 py-2 cursor-pointer rounded-lg transition-colors ${
                  currentSection?.subsection?.id === sub.id
                    ? 'bg-primary/10 text-primary'
                    : 'hover:bg-muted text-text/60 hover:text-text'
                }`}
                style={{ paddingLeft: `${(level + 1) * 1.5 + 0.75}rem` }}
                onClick={() => loadDoc(sub.id, section, sub)}
              >
                <span className="text-xs flex-1">{sub.title}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  if (!docStructure) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-text/60">Loading documentation...</div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-background">
      {/* Sidebar */}
      <div className="w-64 border-r border-border bg-card overflow-y-auto">
        <div className="p-4 border-b border-border">
          <div className="flex items-center gap-2 mb-4">
            <BookOpen className="text-primary" size={24} />
            <h1 className="text-lg font-semibold text-text">Documentation</h1>
          </div>

          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-text/40" size={16} />
            <input
              type="text"
              placeholder="Search docs..."
              className="w-full pl-9 pr-3 py-2 text-sm bg-background border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50 text-text placeholder-text/40"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
        </div>

        <div className="p-3 space-y-1">
          {docStructure.sections.map(section => renderSidebarItem(section))}
        </div>

        <div className="p-3 border-t border-border">
          <button
            onClick={() => onNavigate('dashboard')}
            className="flex items-center gap-2 w-full px-3 py-2 text-sm text-text/70 hover:text-text hover:bg-muted rounded-lg transition-colors"
          >
            <Home size={16} />
            Back to Dashboard
          </button>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto p-8">
          {loading ? (
            <div className="text-text/60">Loading...</div>
          ) : (
            <div className="prose prose-invert max-w-none">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  h1: ({ node, ...props }) => <h1 className="text-3xl font-bold text-text mb-4" {...props} />,
                  h2: ({ node, ...props }) => <h2 className="text-2xl font-semibold text-text mt-8 mb-3" {...props} />,
                  h3: ({ node, ...props }) => <h3 className="text-xl font-semibold text-text mt-6 mb-2" {...props} />,
                  h4: ({ node, ...props }) => <h4 className="text-lg font-semibold text-text mt-4 mb-2" {...props} />,
                  p: ({ node, ...props }) => <p className="text-text/80 mb-4 leading-relaxed" {...props} />,
                  ul: ({ node, ...props }) => <ul className="list-disc list-inside text-text/80 mb-4 space-y-1" {...props} />,
                  ol: ({ node, ...props }) => <ol className="list-decimal list-inside text-text/80 mb-4 space-y-1" {...props} />,
                  li: ({ node, ...props }) => <li className="text-text/80" {...props} />,
                  a: ({ node, ...props }) => <a className="text-primary hover:underline" {...props} />,
                  code: ({ node, inline, ...props }) =>
                    inline ? (
                      <code className="bg-muted text-primary px-1 py-0.5 rounded text-sm" {...props} />
                    ) : (
                      <code className="block bg-muted text-text p-4 rounded-lg overflow-x-auto text-sm" {...props} />
                    ),
                  pre: ({ node, ...props }) => <pre className="bg-muted rounded-lg p-4 mb-4 overflow-x-auto" {...props} />,
                  table: ({ node, ...props }) => <table className="min-w-full border-collapse border border-border mb-4" {...props} />,
                  thead: ({ node, ...props }) => <thead className="bg-muted" {...props} />,
                  tbody: ({ node, ...props }) => <tbody {...props} />,
                  tr: ({ node, ...props }) => <tr className="border-b border-border" {...props} />,
                  th: ({ node, ...props }) => <th className="border border-border px-4 py-2 text-left text-text font-semibold" {...props} />,
                  td: ({ node, ...props }) => <td className="border border-border px-4 py-2 text-text/80" {...props} />,
                  blockquote: ({ node, ...props }) => <blockquote className="border-l-4 border-primary pl-4 italic text-text/70 mb-4" {...props} />,
                  strong: ({ node, ...props }) => <strong className="font-semibold text-text" {...props} />,
                  em: ({ node, ...props }) => <em className="italic text-text/80" {...props} />,
                }}
              >
                {docContent}
              </ReactMarkdown>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
