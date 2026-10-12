"""Investment thesis generator for companies mentioned in podcasts."""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from .llm_client import LLMClient


class ThesisGenerator:
    """Generate company/sector analyses from podcast transcripts for interview prep and education."""

    def __init__(self, db, llm_model: str = "llama3.2:3b",
                 llm_client: LLMClient = None):
        """Initialize thesis generator.

        Args:
            db: Database instance
            llm_model: Model name (used when llm_client not provided)
            llm_client: Shared LLM client with retry/fallback support
        """
        self.db = db
        self.llm_model = llm_model
        self.llm_client = llm_client or LLMClient(
            provider="ollama", model=llm_model
        )

    def get_available_companies(self, lookback_days: int = 90) -> List[str]:
        """Get list of companies mentioned in podcast summaries.

        Args:
            lookback_days: Number of days to look back

        Returns:
            Sorted list of unique company names
        """
        query = """
        SELECT DISTINCT json_extract_string(startup.value, '$') as startup
        FROM summaries s
        JOIN episodes e ON s.episode_id = e.id,
             json_each(s.startups) as startup
        WHERE e.date >= CURRENT_DATE - CAST(? AS INTEGER) * INTERVAL 1 DAY
        AND s.startups IS NOT NULL
        AND json_array_length(s.startups) > 0
        AND json_extract_string(startup.value, '$') IS NOT NULL
        AND json_extract_string(startup.value, '$') != ''
        ORDER BY startup
        """

        results = self.db.conn.execute(query, (int(lookback_days),)).fetchall()
        return [row[0] for row in results if row[0]]

    def get_available_sectors(self, lookback_days: int = 90) -> Dict[str, int]:
        """Get common topics/themes from summaries as sector indicators.

        Args:
            lookback_days: Number of days to look back

        Returns:
            Dictionary of topic -> mention count
        """
        query = """
        SELECT json_extract_string(topic.value, '$') as topic, COUNT(*) as count
        FROM summaries s
        JOIN episodes e ON s.episode_id = e.id,
             json_each(s.key_topics) as topic
        WHERE e.date >= CURRENT_DATE - CAST(? AS INTEGER) * INTERVAL 1 DAY
        AND s.key_topics IS NOT NULL
        AND json_array_length(s.key_topics) > 0
        AND json_extract_string(topic.value, '$') IS NOT NULL
        AND json_extract_string(topic.value, '$') != ''
        GROUP BY 1
        ORDER BY count DESC
        """

        results = self.db.conn.execute(query, (int(lookback_days),)).fetchall()
        return {row[0]: row[1] for row in results}

    def find_peer_companies(self, company: str, lookback_days: int = 90) -> List[str]:
        """Find peer companies that might be related to the search.

        Simple implementation: finds companies mentioned in same episodes.

        Args:
            company: Company to find peers for
            lookback_days: Number of days to look back

        Returns:
            List of peer company names
        """
        # Get episodes that mention any partial match
        query = """
        SELECT DISTINCT json_extract_string(startup.value, '$') as startup
        FROM summaries s
        JOIN episodes e ON s.episode_id = e.id,
             json_each(s.startups) as startup
        WHERE e.id IN (
            SELECT DISTINCT s2.episode_id
            FROM summaries s2
            JOIN episodes e2 ON s2.episode_id = e2.id,
                 json_each(s2.startups) as s2_startup
            WHERE json_extract_string(s2_startup.value, '$') ILIKE ?
            AND e2.date >= CURRENT_DATE - CAST(? AS INTEGER) * INTERVAL 1 DAY
        )
        AND s.startups IS NOT NULL
        AND json_array_length(s.startups) > 0
        AND json_extract_string(startup.value, '$') IS NOT NULL
        AND json_extract_string(startup.value, '$') != ''
        AND json_extract_string(startup.value, '$') NOT ILIKE ?
        ORDER BY startup
        LIMIT 10
        """

        pattern = f"%{company}%"
        results = self.db.conn.execute(query, [pattern, int(lookback_days), pattern]).fetchall()
        return [row[0] for row in results if row[0]]

    def generate_thesis(self, company: str, lookback_days: int = 30) -> Dict[str, Any]:
        """Generate company analysis for interview prep and education.

        Args:
            company: Company name to analyze
            lookback_days: Number of days to look back for mentions

        Returns:
            Dictionary containing analysis and supporting evidence
        """
        # Get all mentions of the company from transcripts
        mentions = self._get_company_mentions(company, lookback_days)
        peer_companies = []
        based_on_peers = False

        if not mentions:
            # Try to find peer companies
            peer_companies = self.find_peer_companies(company, lookback_days)

            if not peer_companies:
                return {
                    "error": f"No mentions of '{company}' or related companies found in the last {lookback_days} days",
                    "suggestion": "Try increasing --lookback-days or check available companies with p3 thesis --list-companies"
                }

            # Get mentions for peer companies instead
            based_on_peers = True
            for peer in peer_companies[:3]:  # Use top 3 peers
                peer_mentions = self._get_company_mentions(peer, lookback_days)
                mentions.extend(peer_mentions)

        # Extract context around mentions
        context = self._extract_context(mentions)

        # Get related summaries for additional insights
        summaries = self._get_related_summaries(company, lookback_days)

        if based_on_peers:
            # Also get summaries for peer companies
            for peer in peer_companies[:3]:
                peer_summaries = self._get_related_summaries(peer, lookback_days)
                summaries.extend(peer_summaries)

        # Generate analysis using LLM
        thesis = self._generate_company_analysis(
            company,
            context,
            summaries,
            based_on_peers=based_on_peers,
            peer_companies=peer_companies
        )

        return {
            "company": company,
            "generated_at": datetime.now().isoformat(),
            "lookback_days": lookback_days,
            "mention_count": len(mentions),
            "episode_count": len(set(m['episode_id'] for m in mentions)),
            "based_on_peers": based_on_peers,
            "peer_companies": peer_companies if based_on_peers else [],
            "thesis": thesis,
            "sources": self._format_sources(mentions)
        }

    def generate_sector_thesis(self, sector: str, lookback_days: int = 30) -> Dict[str, Any]:
        """Generate investment thesis for a sector/theme.

        Args:
            sector: Sector or theme to analyze (e.g., "AI", "SaaS", "crypto")
            lookback_days: Number of days to look back

        Returns:
            Dictionary containing sector thesis
        """
        # Get transcripts mentioning the sector
        mentions = self._search_transcripts(sector, lookback_days)

        if not mentions:
            return {
                "error": f"No mentions of '{sector}' found in the last {lookback_days} days"
            }

        # Extract context
        context = self._extract_context(mentions)

        # Get related summaries
        summaries = self._get_sector_summaries(sector, lookback_days)

        # Generate sector thesis
        thesis = self._generate_sector_analysis(sector, context, summaries)

        return {
            "sector": sector,
            "generated_at": datetime.now().isoformat(),
            "lookback_days": lookback_days,
            "mention_count": len(mentions),
            "episode_count": len(set(m['episode_id'] for m in mentions)),
            "thesis": thesis,
            "sources": self._format_sources(mentions)
        }

    def _get_company_mentions(self, company: str, lookback_days: int) -> List[Dict[str, Any]]:
        """Get all transcript mentions of a company."""
        # First, get episodes that have transcripts mentioning the company
        # Since transcripts are stored as segments, we need to aggregate them
        query = """
        SELECT DISTINCT
            e.id as episode_id,
            e.title as episode_title,
            e.date,
            p.title as podcast_title
        FROM transcripts t
        JOIN episodes e ON t.episode_id = e.id
        JOIN podcasts p ON e.podcast_id = p.id
        WHERE t.text ILIKE ?
        AND e.date >= CURRENT_DATE - CAST(? AS INTEGER) * INTERVAL 1 DAY
        ORDER BY e.date DESC
        """

        results = self.db.conn.execute(
            query,
            [f"%{company}%", int(lookback_days)]
        ).fetchall()

        mentions = []
        for row in results:
            episode_id = row[0]

            # Get all transcript segments for this episode and combine them
            segments = self.db.get_transcripts_for_episode(episode_id)
            full_text = "\n".join(segment['text'] for segment in segments)

            mentions.append({
                "episode_id": episode_id,
                "full_text": full_text,
                "episode_title": row[1],
                "published_date": row[2],
                "podcast_name": row[3]
            })

        return mentions

    def _search_transcripts(self, keyword: str, lookback_days: int) -> List[Dict[str, Any]]:
        """Search transcripts for keyword/phrase."""
        # Same implementation as _get_company_mentions but more generic
        return self._get_company_mentions(keyword, lookback_days)

    def _extract_context(self, mentions: List[Dict[str, Any]]) -> str:
        """Extract relevant context around company mentions."""
        contexts = []

        for mention in mentions:
            full_text = mention['full_text']
            podcast = mention['podcast_name']
            title = mention['episode_title']
            date = mention['published_date']

            # For now, include the full transcript
            # In production, you might want to extract paragraphs around mentions
            contexts.append(f"""
Source: {podcast} - {title} ({date})
Content: {full_text[:5000]}...
""")

        return "\n\n---\n\n".join(contexts)

    def _get_related_summaries(self, company: str, lookback_days: int) -> List[Dict[str, Any]]:
        """Get summaries that mention the company."""
        query = """
        SELECT
            s.key_topics,
            s.themes,
            s.key_insights,
            s.actionable_takeaways,
            s.full_summary,
            s.startups,
            e.title,
            p.title as podcast_name
        FROM summaries s
        JOIN episodes e ON s.episode_id = e.id
        JOIN podcasts p ON e.podcast_id = p.id
        WHERE s.startups::TEXT ILIKE ?
        AND e.date >= CURRENT_DATE - CAST(? AS INTEGER) * INTERVAL 1 DAY
        ORDER BY e.date DESC
        """

        results = self.db.conn.execute(
            query,
            [f"%{company}%", int(lookback_days)]
        ).fetchall()

        summaries = []
        for row in results:
            summaries.append({
                "key_topics": json.loads(row[0]) if row[0] else [],
                "themes": json.loads(row[1]) if row[1] else [],
                "key_insights": json.loads(row[2]) if row[2] else [],
                "actionable_takeaways": json.loads(row[3]) if row[3] else [],
                "full_summary": row[4],
                "startups": json.loads(row[5]) if row[5] else [],
                "episode_title": row[6],
                "podcast_name": row[7]
            })

        return summaries

    def _get_sector_summaries(self, sector: str, lookback_days: int) -> List[Dict[str, Any]]:
        """Get summaries related to a sector."""
        query = """
        SELECT
            s.key_topics,
            s.themes,
            s.key_insights,
            s.actionable_takeaways,
            s.full_summary,
            e.title,
            p.title as podcast_name
        FROM summaries s
        JOIN episodes e ON s.episode_id = e.id
        JOIN podcasts p ON e.podcast_id = p.id
        WHERE (
            s.key_topics::TEXT ILIKE ? OR
            s.themes::TEXT ILIKE ? OR
            s.full_summary ILIKE ?
        )
        AND e.date >= CURRENT_DATE - CAST(? AS INTEGER) * INTERVAL 1 DAY
        ORDER BY e.date DESC
        """

        pattern = f"%{sector}%"
        results = self.db.conn.execute(
            query,
            [pattern, pattern, pattern, int(lookback_days)]
        ).fetchall()

        summaries = []
        for row in results:
            summaries.append({
                "key_topics": json.loads(row[0]) if row[0] else [],
                "themes": json.loads(row[1]) if row[1] else [],
                "key_insights": json.loads(row[2]) if row[2] else [],
                "actionable_takeaways": json.loads(row[3]) if row[3] else [],
                "full_summary": row[4],
                "episode_title": row[5],
                "podcast_name": row[6]
            })

        return summaries

    def _generate_company_analysis(self, company: str, context: str,
                                   summaries: List[Dict[str, Any]],
                                   based_on_peers: bool = False,
                                   peer_companies: List[str] = None) -> Dict[str, Any]:
        """Generate company analysis for interview prep and education."""
        # Prepare summary insights
        summary_text = ""
        for s in summaries[:5]:  # Limit to 5 most recent
            summary_text += f"\nPodcast: {s['podcast_name']} - {s['episode_title']}\n"
            summary_text += f"Summary: {s['full_summary']}\n"
            if s['key_insights']:
                summary_text += f"Insights: {', '.join(s['key_insights'])}\n"

        peer_context = ""
        if based_on_peers and peer_companies:
            peer_context = f"\n\nIMPORTANT: {company} was not directly mentioned in podcasts. This analysis is based on discussions about peer companies: {', '.join(peer_companies[:3])}. Use these insights to infer what might be relevant to {company}."

        prompt = f"""You are preparing someone for a job interview or helping them learn about companies in tech/VC.
Generate a comprehensive company analysis for {company} based ONLY on the podcast transcripts and summaries provided below.{peer_context}

CRITICAL: You must ONLY use information from the provided sources below. Do NOT use your general knowledge or training data.
For every claim, insight, or data point, cite the specific podcast and episode it came from using the format "(Source: Podcast Name - Episode Title)".
If the provided sources do not contain enough information about a topic, say "Not discussed in available sources" rather than filling in from general knowledge.

FULL TRANSCRIPT CONTEXT:
{context[:15000]}

STRUCTURED SUMMARIES:
{summary_text}

Generate a detailed analysis in JSON format optimized for interview preparation and education:

{{
  "executive_summary": "2-3 sentence overview citing which podcasts discussed this company",
  "what_they_do": "Clear explanation based on podcast discussions, citing sources",
  "why_its_interesting": ["reason 1 (Source: Podcast - Episode)", "reason 2 (Source: ...)", "reason 3"],
  "potential_concerns": ["concern 1 (Source: ...)", "concern 2", "concern 3"],
  "market_context": "Industry dynamics as discussed in the podcasts, with source citations",
  "key_talking_points": ["point 1 for interviews (Source: ...)", "point 2", "point 3"],
  "recent_developments": ["development 1 (Source: ...)", "development 2"],
  "growth_drivers": ["driver 1 (Source: ...)", "driver 2", "driver 3"],
  "challenges": ["challenge 1 (Source: ...)", "challenge 2", "challenge 3"],
  "competitive_differentiation": "What makes this company unique, as discussed in podcasts",
  "interview_angles": ["angle 1 to discuss in interviews", "angle 2", "angle 3"],
  "learning_takeaways": ["takeaway 1", "takeaway 2", "takeaway 3"],
  "confidence_level": "HIGH/MEDIUM/LOW based on how much direct discussion exists in the provided sources"
}}

Focus on insights that help someone speak intelligently about the company in interviews or conversations. Be educational rather than prescriptive about investment decisions. Always cite the specific podcast source."""

        try:
            result = self.llm_client.chat_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=3000
            )

            if result:
                return result

            return {
                "error": "LLM returned no response",
                "executive_summary": "Unable to generate thesis - no LLM response"
            }

        except Exception as e:
            return {
                "error": f"Failed to generate thesis: {str(e)}",
                "executive_summary": "Unable to generate thesis due to processing error"
            }

    def _generate_sector_analysis(self, sector: str, context: str,
                                  summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate sector/theme analysis using LLM."""
        # Prepare summary insights
        summary_text = ""
        all_insights = []
        all_themes = []

        for s in summaries[:10]:  # More summaries for sector analysis
            summary_text += f"\nPodcast: {s['podcast_name']} - {s['episode_title']}\n"
            summary_text += f"Summary: {s['full_summary']}\n"
            all_insights.extend(s['key_insights'])
            all_themes.extend(s['themes'])

        prompt = f"""You are an experienced venture capital analyst. Generate a comprehensive sector analysis for {sector} based ONLY on the podcast transcripts and summaries provided below.

CRITICAL: You must ONLY use information from the provided sources below. Do NOT use your general knowledge or training data.
For every claim, trend, or data point, cite the specific podcast and episode it came from using the format "(Source: Podcast Name - Episode Title)".
If the provided sources do not contain enough information about a topic, say "Not discussed in available sources" rather than filling in from general knowledge.

FULL TRANSCRIPT CONTEXT:
{context[:15000]}

STRUCTURED SUMMARIES AND INSIGHTS:
{summary_text}

Key Themes Mentioned: {', '.join(set(all_themes))}
Key Insights: {', '.join(set(all_insights))}

Generate a detailed sector thesis in JSON format:

{{
  "executive_summary": "2-3 sentence overview citing which podcasts discussed this sector",
  "market_trends": ["trend 1 (Source: Podcast - Episode)", "trend 2 (Source: ...)", "trend 3"],
  "key_players": ["company 1 (Source: ...)", "company 2", "company 3"],
  "investment_opportunities": ["opportunity 1 (Source: ...)", "opportunity 2", "opportunity 3"],
  "headwinds": ["challenge 1 (Source: ...)", "challenge 2"],
  "tailwinds": ["driver 1 (Source: ...)", "driver 2", "driver 3"],
  "time_horizon": "When this opportunity is likely to materialize, based on source discussions",
  "capital_intensity": "Assessment of capital requirements as discussed in sources",
  "recommended_approach": "How to approach investing in this sector based on podcast insights",
  "watch_list": ["specific companies or indicators mentioned in podcasts"],
  "confidence_level": "HIGH/MEDIUM/LOW based on how much direct discussion exists in the provided sources",
  "contrarian_view": "What the market might be missing, as suggested by podcast discussions"
}}

Focus on actionable insights for allocating capital in this sector. Always cite the specific podcast source."""

        try:
            result = self.llm_client.chat_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=3000
            )

            if result:
                return result

            return {
                "error": "LLM returned no response",
                "executive_summary": "Unable to generate sector analysis - no LLM response"
            }

        except Exception as e:
            return {
                "error": f"Failed to generate sector analysis: {str(e)}",
                "executive_summary": "Unable to generate analysis due to processing error"
            }

    def _format_sources(self, mentions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Format source citations."""
        sources = []
        for m in mentions:
            sources.append({
                "podcast": m['podcast_name'],
                "episode": m['episode_title'],
                "date": str(m['published_date'])
            })
        return sources

    def save_thesis(self, thesis_data: Dict[str, Any]) -> Path:
        """Save thesis to markdown file."""
        output_dir = Path("outputs/thesis")
        output_dir.mkdir(parents=True, exist_ok=True)

        if "company" in thesis_data:
            subject = thesis_data["company"]
            thesis_type = "Company"
        else:
            subject = thesis_data.get("sector", "Unknown")
            thesis_type = "Sector"

        filename = output_dir / f"{subject.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        # Format markdown
        content = self._format_thesis_markdown(thesis_data, thesis_type)

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)

        return Path(filename)

    def _format_thesis_markdown(self, data: Dict[str, Any], thesis_type: str) -> str:
        """Format analysis as markdown."""
        subject = data.get("company") or data.get("sector", "Unknown")
        thesis = data.get("thesis", {})

        if "error" in thesis:
            return f"# {thesis_type} Analysis: {subject}\n\nError: {thesis['error']}"

        # Check if based on peer companies
        peer_note = ""
        if data.get("based_on_peers") and data.get("peer_companies"):
            peers = ", ".join(data.get("peer_companies", [])[:3])
            peer_note = f"\n**⚠️ Note:** {subject} was not directly mentioned. Analysis based on peer companies: {peers}"

        md = f"""# {thesis_type} Analysis: {subject}
*For Interview Preparation and Education*

**Generated:** {data.get('generated_at', 'N/A')}
**Analysis Period:** Last {data.get('lookback_days', 'N/A')} days
**Sources:** {data.get('mention_count', 0)} mentions across {data.get('episode_count', 0)} episodes
**Confidence:** {thesis.get('confidence_level', 'N/A')}{peer_note}

---

## Executive Summary

{thesis.get('executive_summary', 'N/A')}

"""

        if thesis_type == "Company":
            md += f"""
## What They Do

{thesis.get('what_they_do', 'N/A')}

## Why It's Interesting

{self._format_list(thesis.get('why_its_interesting', []))}

## Key Talking Points (For Interviews)

{self._format_list(thesis.get('key_talking_points', []))}

## Interview Angles

{self._format_list(thesis.get('interview_angles', []))}

## Market Context

{thesis.get('market_context', 'N/A')}

## Recent Developments

{self._format_list(thesis.get('recent_developments', []))}

## Growth Drivers

{self._format_list(thesis.get('growth_drivers', []))}

## Competitive Differentiation

{thesis.get('competitive_differentiation', 'N/A')}

## Potential Concerns

{self._format_list(thesis.get('potential_concerns', []))}

## Challenges

{self._format_list(thesis.get('challenges', []))}

## Learning Takeaways

{self._format_list(thesis.get('learning_takeaways', []))}
"""
        else:  # Sector analysis
            md += f"""
## Market Trends

{self._format_list(thesis.get('market_trends', []))}

## Key Players

{self._format_list(thesis.get('key_players', []))}

## Investment Opportunities

{self._format_list(thesis.get('investment_opportunities', []))}

## Tailwinds

{self._format_list(thesis.get('tailwinds', []))}

## Headwinds

{self._format_list(thesis.get('headwinds', []))}

## Time Horizon

{thesis.get('time_horizon', 'N/A')}

## Capital Intensity

{thesis.get('capital_intensity', 'N/A')}

## Recommended Approach

{thesis.get('recommended_approach', 'N/A')}

## Watch List

{self._format_list(thesis.get('watch_list', []))}

## Contrarian View

{thesis.get('contrarian_view', 'N/A')}
"""

        # Add sources
        md += "\n---\n\n## Sources\n\n"
        for source in data.get('sources', []):
            md += f"- **{source['podcast']}**: {source['episode']} ({source['date']})\n"

        return md

    def _format_list(self, items: List[str]) -> str:
        """Format list items as markdown."""
        if not items:
            return "- None identified"
        return "\n".join(f"- {item}" for item in items)

