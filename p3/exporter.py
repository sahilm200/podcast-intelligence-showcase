"""Export functionality for P³ digests."""

import json
from datetime import date
from typing import Dict, List, Any


class DigestExporter:
    def __init__(self, db):
        self.db = db

    def export_markdown(self, summaries: List[Dict[str, Any]], target_date: date,
                       include_newsletters: bool = True) -> str:
        """Export summaries as Markdown."""
        content = [f"# Podcast Digest - {target_date}\n"]

        if not summaries:
            content.append("No podcast summaries available for this date.\n")
        
        # Group by podcast
        by_podcast = {}
        for summary in summaries:
            podcast = summary['podcast_title']
            if podcast not in by_podcast:
                by_podcast[podcast] = []
            by_podcast[podcast].append(summary)
        
        for podcast_name, episodes in by_podcast.items():
            content.append(f"## {podcast_name}\n")
            
            for episode in episodes:
                content.append(f"### {episode['episode_title']}\n")
                
                if episode['full_summary']:
                    content.append(f"**Summary:** {episode['full_summary']}\n")
                
                if episode['key_topics']:
                    content.append("**Key Topics:**")
                    for i, topic in enumerate(episode['key_topics'], 1):
                        content.append(f"{i}. {topic}")
                    content.append("")

                if episode['themes']:
                    content.append("**Themes:**")
                    for i, theme in enumerate(episode['themes'], 1):
                        content.append(f"{i}. {theme}")
                    content.append("")

                if episode['quotes']:
                    content.append("**Notable Quotes:**")
                    for i, quote in enumerate(episode['quotes'], 1):
                        content.append(f"{i}. > {quote}")
                    content.append("")

                if episode['startups']:
                    content.append("**Companies/Startups Mentioned:**")
                    for i, startup in enumerate(episode['startups'], 1):
                        content.append(f"{i}. {startup}")
                    content.append("")

                if episode.get('key_insights'):
                    content.append("**Key Insights:**")
                    for i, insight in enumerate(episode['key_insights'], 1):
                        content.append(f"{i}. {insight}")
                    content.append("")

                if episode.get('actionable_takeaways'):
                    content.append("**Actionable Takeaways:**")
                    for i, takeaway in enumerate(episode['actionable_takeaways'], 1):
                        content.append(f"{i}. {takeaway}")
                    content.append("")

                content.append("---\n")

        # Add newsletter summaries if enabled
        if include_newsletters:
            newsletter_summaries = self.db.get_newsletter_summaries_by_date(target_date)
            if newsletter_summaries:
                content.append("\n# 📰 Newsletter Insights\n")

                # Group by newsletter
                by_newsletter = {}
                for summary in newsletter_summaries:
                    newsletter = summary['newsletter_name']
                    if newsletter not in by_newsletter:
                        by_newsletter[newsletter] = []
                    by_newsletter[newsletter].append(summary)

                for newsletter_name, articles in by_newsletter.items():
                    content.append(f"## {newsletter_name}\n")

                    for article in articles:
                        content.append(f"### {article['article_title']}\n")
                        content.append(f"🔗 [Read Article]({article['article_url']})\n")

                        if article['full_summary']:
                            content.append(f"**Summary:** {article['full_summary']}\n")

                        if article['key_topics']:
                            content.append("**Key Topics:**")
                            for i, topic in enumerate(article['key_topics'], 1):
                                content.append(f"{i}. {topic}")
                            content.append("")

                        if article['themes']:
                            content.append("**Themes:**")
                            for i, theme in enumerate(article['themes'], 1):
                                content.append(f"{i}. {theme}")
                            content.append("")

                        if article['quotes']:
                            content.append("**Notable Quotes:**")
                            for i, quote in enumerate(article['quotes'], 1):
                                content.append(f"{i}. > {quote}")
                            content.append("")

                        if article['companies']:
                            content.append("**Companies Mentioned:**")
                            for i, company in enumerate(article['companies'], 1):
                                content.append(f"{i}. {company}")
                            content.append("")

                        if article.get('key_insights'):
                            content.append("**Key Insights:**")
                            for i, insight in enumerate(article['key_insights'], 1):
                                content.append(f"{i}. {insight}")
                            content.append("")

                        if article.get('actionable_takeaways'):
                            content.append("**Actionable Takeaways:**")
                            for i, takeaway in enumerate(article['actionable_takeaways'], 1):
                                content.append(f"{i}. {takeaway}")
                            content.append("")

                        content.append("---\n")

        return "\n".join(content)

    def export_json(self, summaries: List[Dict[str, Any]], target_date: date,
                   include_newsletters: bool = True) -> str:
        """Export summaries as JSON."""
        export_data = {
            "date": str(target_date),
            "total_episodes": len(summaries),
            "summaries": summaries
        }

        if include_newsletters:
            newsletter_summaries = self.db.get_newsletter_summaries_by_date(target_date)
            export_data["total_newsletters"] = len(newsletter_summaries)
            export_data["newsletter_summaries"] = newsletter_summaries

        return json.dumps(export_data, indent=2, default=str)

    def export_email_html(self, summaries: List[Dict[str, Any]], target_date: date,
                          include_newsletters: bool = True) -> str:
        """Export summaries as HTML for email."""
        html = f"""
        <html>
        <head>
            <title>Podcast Digest - {target_date}</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; }}
                h1, h2, h3 {{ color: #333; }}
                .podcast {{ margin-bottom: 2em; }}
                .episode {{ margin-bottom: 1.5em; padding: 1em; background: #f9f9f9; }}
                .summary {{ font-style: italic; margin-bottom: 1em; }}
                .topics, .themes, .startups, .insights, .takeaways {{ margin-bottom: 0.5em; }}
                .quote {{ background: #e8e8e8; padding: 0.5em; margin: 0.5em 0; border-left: 3px solid #666; }}
                .insights {{ color: #0066cc; }}
                .takeaways {{ color: #009900; }}
                ol, ul {{ margin: 0.5em 0; padding-left: 2em; }}
                ol {{ list-style-type: decimal; }}
            </style>
        </head>
        <body>
            <h1>Podcast Digest - {target_date}</h1>
        """
        
        if not summaries:
            html += "<p>No summaries available for this date.</p>"
        else:
            # Group by podcast
            by_podcast = {}
            for summary in summaries:
                podcast = summary['podcast_title']
                if podcast not in by_podcast:
                    by_podcast[podcast] = []
                by_podcast[podcast].append(summary)
            
            for podcast_name, episodes in by_podcast.items():
                html += f'<div class="podcast"><h2>{podcast_name}</h2>'
                
                for episode in episodes:
                    html += f'<div class="episode"><h3>{episode["episode_title"]}</h3>'
                    
                    if episode['full_summary']:
                        html += f'<div class="summary"><strong>Summary:</strong> {episode["full_summary"]}</div>'
                    
                    if episode['key_topics']:
                        html += '<div class="topics"><strong>Key Topics:</strong><ol>'
                        for topic in episode['key_topics']:
                            html += f'<li>{topic}</li>'
                        html += '</ol></div>'

                    if episode['themes']:
                        html += '<div class="themes"><strong>Themes:</strong><ol>'
                        for theme in episode['themes']:
                            html += f'<li>{theme}</li>'
                        html += '</ol></div>'

                    if episode['quotes']:
                        html += '<div><strong>Notable Quotes:</strong><ol>'
                        for quote in episode['quotes']:
                            html += f'<li><div class="quote">{quote}</div></li>'
                        html += '</ol></div>'

                    if episode['startups']:
                        html += '<div class="startups"><strong>Companies/Startups:</strong><ol>'
                        for startup in episode['startups']:
                            html += f'<li>{startup}</li>'
                        html += '</ol></div>'

                    if episode.get('key_insights'):
                        html += '<div class="insights"><strong>Key Insights:</strong><ol>'
                        for insight in episode['key_insights']:
                            html += f'<li>{insight}</li>'
                        html += '</ol></div>'

                    if episode.get('actionable_takeaways'):
                        html += '<div class="takeaways"><strong>Actionable Takeaways:</strong><ol>'
                        for takeaway in episode['actionable_takeaways']:
                            html += f'<li>{takeaway}</li>'
                        html += '</ol></div>'

                    html += '</div>'  # episode
                
                html += '</div>'  # podcast

        # Add newsletter summaries if enabled
        if include_newsletters:
            newsletter_summaries = self.db.get_newsletter_summaries_by_date(target_date)
            if newsletter_summaries:
                html += '<h1 style="margin-top: 2em; border-top: 2px solid #333; padding-top: 1em;">📰 Newsletter Insights</h1>'

                # Group by newsletter
                by_newsletter = {}
                for summary in newsletter_summaries:
                    newsletter = summary['newsletter_name']
                    if newsletter not in by_newsletter:
                        by_newsletter[newsletter] = []
                    by_newsletter[newsletter].append(summary)

                for newsletter_name, articles in by_newsletter.items():
                    html += f'<div class="podcast"><h2>{newsletter_name}</h2>'

                    for article in articles:
                        html += f'<div class="episode"><h3>{article["article_title"]}</h3>'
                        html += f'<p>🔗 <a href="{article["article_url"]}">Read Article</a></p>'

                        if article['full_summary']:
                            html += f'<div class="summary"><strong>Summary:</strong> {article["full_summary"]}</div>'

                        if article['key_topics']:
                            html += '<div class="topics"><strong>Key Topics:</strong><ol>'
                            for topic in article['key_topics']:
                                html += f'<li>{topic}</li>'
                            html += '</ol></div>'

                        if article['themes']:
                            html += '<div class="themes"><strong>Themes:</strong><ol>'
                            for theme in article['themes']:
                                html += f'<li>{theme}</li>'
                            html += '</ol></div>'

                        if article['quotes']:
                            html += '<div><strong>Notable Quotes:</strong><ol>'
                            for quote in article['quotes']:
                                html += f'<li><div class="quote">{quote}</div></li>'
                            html += '</ol></div>'

                        if article['companies']:
                            html += '<div class="startups"><strong>Companies:</strong><ol>'
                            for company in article['companies']:
                                html += f'<li>{company}</li>'
                            html += '</ol></div>'

                        if article.get('key_insights'):
                            html += '<div class="insights"><strong>Key Insights:</strong><ol>'
                            for insight in article['key_insights']:
                                html += f'<li>{insight}</li>'
                            html += '</ol></div>'

                        if article.get('actionable_takeaways'):
                            html += '<div class="takeaways"><strong>Actionable Takeaways:</strong><ol>'
                            for takeaway in article['actionable_takeaways']:
                                html += f'<li>{takeaway}</li>'
                            html += '</ol></div>'

                        html += '</div>'  # episode

                    html += '</div>'  # podcast

        html += """
        </body>
        </html>
        """

        return html

