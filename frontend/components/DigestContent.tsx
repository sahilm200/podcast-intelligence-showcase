'use client'

import ReactMarkdown from 'react-markdown'

export default function DigestContent({ content }: { content: string }) {
  return (
    <article className="prose prose-sm max-w-none
      prose-headings:text-[#1d1d1f] prose-headings:font-semibold prose-headings:tracking-tight
      prose-h1:text-[22px] prose-h1:mb-5
      prose-h2:text-[17px] prose-h2:mt-8 prose-h2:mb-2
      prose-h3:text-[15px] prose-h3:mt-6 prose-h3:mb-1.5
      prose-p:text-[#3a3a3c] prose-p:leading-relaxed prose-p:text-[15px]
      prose-strong:text-[#1d1d1f] prose-strong:font-semibold
      prose-em:text-[#6e6e73]
      prose-li:text-[#3a3a3c] prose-li:text-[15px]
      prose-hr:border-[rgba(0,0,0,0.08)]
      prose-a:text-[#0071e3] prose-a:no-underline hover:prose-a:underline">
      <ReactMarkdown>{content}</ReactMarkdown>
    </article>
  )
}

