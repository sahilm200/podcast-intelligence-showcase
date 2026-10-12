"""Blog post generation with AP English teacher grading system.

Inspired by Tomasz Tunguz's innovative approach to AI-assisted writing
with iterative grading and improvement loops.
"""

import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from .database import P3Database
from .llm_client import LLMClient


class BlogWriter:
    def __init__(self, db: P3Database, llm_provider: str = "ollama",
                 llm_model: str = "llama3.2:latest", target_grade: float = 91.0,
                 llm_client: LLMClient = None):
        self.db = db
        self.llm_provider = llm_provider.lower()
        self.llm_model = llm_model
        self.target_grade = target_grade
        self.max_iterations = 3
        self.llm_client = llm_client or LLMClient(
            provider=llm_provider, model=llm_model
        )
        
    def generate_blog_post_from_digest(self, topic: str, digest_data,
                                     context_posts: List[str] = None,
                                     extra_instructions: str = None) -> Dict[str, Any]:
        """Generate blog post from podcast digest(s) with iterative AP English grading.

        Args:
            topic: The main topic/angle for the blog post
            digest_data: Either a single Dict or List of Dicts with structured digest data
            context_posts: Optional list of related blog posts for style matching

        Returns:
            Dict containing final blog post, grades, and iterations
        """

        # Handle both single dict and list of dicts
        if isinstance(digest_data, dict):
            digest_list = [digest_data]
        else:
            digest_list = digest_data

        # Combine data from all episodes
        all_topics = []
        all_themes = []
        all_quotes = []
        all_companies = []
        all_insights = []
        all_takeaways = []
        episode_summaries = []
        source_episodes = []

        for digest in digest_list:
            episode_title = digest.get('episode_title', '')
            podcast_title = digest.get('podcast_title', '')
            summary = digest.get('full_summary', '')

            # Track source episodes
            source_episodes.append({
                'episode': episode_title,
                'podcast': podcast_title
            })

            # Add episode summary with source
            if summary:
                episode_summaries.append(f"**{podcast_title} - {episode_title}**: {summary}")

            # Combine all data
            all_topics.extend(digest.get('key_topics', []))
            all_themes.extend(digest.get('themes', []))
            # Attribute quotes to their source
            source_label = f"{podcast_title}: {episode_title}" if podcast_title and episode_title else "Unknown Source"
            for q in digest.get('quotes', []):
                all_quotes.append(f"{q} -- {source_label}")
            all_companies.extend(digest.get('startups', []))
            all_insights.extend(digest.get('key_insights', []))
            all_takeaways.extend(digest.get('actionable_takeaways', []))

        # Deduplicate while preserving order
        def dedupe(lst):
            seen = set()
            result = []
            for item in lst:
                if item.lower() not in seen:
                    seen.add(item.lower())
                    result.append(item)
            return result

        all_topics = dedupe(all_topics)
        all_themes = dedupe(all_themes)
        all_companies = dedupe(all_companies)

        # Build rich context for the blog post
        context = f"""
        Source Episodes ({len(digest_list)}):
        {chr(10).join(episode_summaries)}

        Combined Key Topics: {', '.join(all_topics[:15])}
        Combined Themes: {', '.join(all_themes[:10])}

        Key Insights from Episodes:
        {chr(10).join('- ' + insight for insight in all_insights[:12])}

        Actionable Takeaways:
        {chr(10).join('- ' + takeaway for takeaway in all_takeaways[:10])}

        Notable Quotes (with source attribution):
        {chr(10).join('- "' + quote + '"' for quote in all_quotes[:15])}

        Companies/Startups Mentioned: {', '.join(all_companies[:20])}
        """
        
        iterations = []
        current_post = ""
        
        # Generate initial blog post
        initial_prompt = self._build_writing_prompt(topic, context, context_posts, extra_instructions)
        current_post = self._generate_with_llm(initial_prompt)
        
        # Iterative grading and improvement (inspired by Tunguz's approach)
        for iteration in range(self.max_iterations):
            grade_result = self._grade_blog_post(current_post)
            iterations.append({
                'iteration': iteration + 1,
                'post': current_post,
                'grade': grade_result['grade'],
                'score': grade_result['score'],
                'feedback': grade_result['feedback']
            })
            
            # Check if we've reached target grade
            if grade_result['score'] >= self.target_grade:
                break
                
            # Improve based on feedback
            if iteration < self.max_iterations - 1:
                improvement_prompt = self._build_improvement_prompt(
                    current_post, grade_result['feedback']
                )
                current_post = self._generate_with_llm(improvement_prompt)
        
        # Generate SEO-friendly slug
        slug = self._generate_slug(topic)

        return {
            'final_post': current_post,
            'final_grade': iterations[-1]['grade'],
            'final_score': iterations[-1]['score'],
            'iterations': iterations,
            'topic': topic,
            'slug': slug,
            'metadata': {
                'source_episodes': source_episodes,
                'episode_count': len(digest_list),
                'generated_at': datetime.now().isoformat(),
                'model_used': self.llm_model
            }
        }
    
    def _build_writing_prompt(self, topic: str, context: str, context_posts: List[str] = None,
                              extra_instructions: str = None) -> str:
        """Build the initial writing prompt based on Tunguz's style guidelines."""

        style_guidelines = """
        Style Guidelines (inspired by Tomasz Tunguz's approach):
        - 500 words or less (49 seconds with reader)
        - No section headers (headers hurt dwell time)
        - Flowing paragraphs that transition smoothly
        - Limit each paragraph to at most two long sentences
        - Strong hook in first few sentences
        - Conclusion that ties back to opening
        - Focus on actionable insights
        - Include specific examples and quotes when relevant
        - When synthesizing multiple sources, find common themes and contrasting viewpoints
        """

        context_section = ""
        if context_posts:
            context_section = f"""
            Related Content for Style Reference:
            {chr(10).join(context_posts[:3])}
            """

        base_prompt = f"""You are an expert blog writer specializing in technology and business content.

        {style_guidelines}

        Topic: {topic}

        Source Material (from multiple podcast episodes):
        {context}

        {context_section}

        Write a compelling blog post that:
        1. Opens with a strong hook that draws readers in
        2. Synthesizes insights from multiple podcast sources into a cohesive narrative
        3. Identifies common themes and patterns across the episodes
        4. Provides actionable takeaways for business/tech readers
        5. Includes relevant quotes to support key points
        6. Weaves together different perspectives where applicable
        7. Concludes with a thought-provoking statement that ties back to the opening

        Important: Don't just list insights from each podcast separately. Instead, weave them
        together into a unified narrative that explores the topic from multiple angles.

        Remember: Be concise, engaging, and focused on delivering value quickly.
        """

        if extra_instructions:
            base_prompt += f"""
        Additional Instructions from the author:
        {extra_instructions}
        """

        return base_prompt
    
    def _grade_blog_post(self, blog_post: str) -> Dict[str, Any]:
        """Grade blog post like an AP English teacher (Tunguz's innovation)."""
        
        grading_prompt = f"""You are an experienced AP English teacher grading a blog post. 
        
        Evaluate this blog post and provide:
        1. Letter grade (A+, A, A-, B+, B, B-, C+, C, C-, D+, D, F)
        2. Numerical score (0-100)
        3. Detailed feedback on each criterion
        
        Evaluation Criteria:
        - Hook/Opening (20 points): Does it grab attention immediately?
        - Argument Clarity (20 points): Is the main point clear and well-supported?
        - Evidence and Examples (20 points): Are quotes and examples used effectively?
        - Paragraph Structure (20 points): Do paragraphs flow smoothly with good transitions?
        - Conclusion Strength (20 points): Does it tie back and leave lasting impact?
        - Overall Engagement (bonus/penalty): Would readers stay engaged throughout?
        
        Blog Post to Grade:
        {blog_post}
        
        Format your response as:
        GRADE: [Letter Grade]
        SCORE: [Numerical Score]
        FEEDBACK: [Detailed feedback with specific suggestions for improvement]
        """
        
        response = self._generate_with_llm(grading_prompt)
        
        # Parse response
        grade_match = re.search(r'GRADE:\s*([A-F][+-]?)', response)
        score_match = re.search(r'SCORE:\s*(\d+)', response)
        feedback_match = re.search(r'FEEDBACK:\s*(.*)', response, re.DOTALL)
        
        return {
            'grade': grade_match.group(1) if grade_match else 'C',
            'score': float(score_match.group(1)) if score_match else 75.0,
            'feedback': feedback_match.group(1).strip() if feedback_match else response,
            'raw_response': response
        }
    
    def _build_improvement_prompt(self, current_post: str, feedback: str) -> str:
        """Build prompt to improve blog post based on feedback."""
        
        return f"""You are revising a blog post based on AP English teacher feedback.
        
        Current Blog Post:
        {current_post}
        
        Teacher Feedback:
        {feedback}
        
        Please rewrite the blog post incorporating the feedback while maintaining:
        - The core message and insights
        - Concise, engaging style (500 words or less)
        - Strong hook and conclusion
        - Smooth paragraph transitions
        - Actionable takeaways
        
        Focus especially on addressing the specific issues mentioned in the feedback.
        """
    
    def _generate_slug(self, topic: str) -> str:
        """Generate URL-friendly slug from topic."""
        # Convert to lowercase and replace spaces/special chars with hyphens
        slug = re.sub(r'[^\w\s-]', '', topic.lower())
        slug = re.sub(r'[-\s]+', '-', slug)
        return slug.strip('-')
    
    def _generate_with_llm(self, prompt: str) -> str:
        """Generate text using configured LLM with retry/fallback."""
        result = self.llm_client.chat(
            messages=[
                {"role": "system", "content": "You are an expert blog writer and writing instructor."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return result if result else "Error: LLM returned no response"
    
    def save_blog_post(self, blog_result: Dict[str, Any], output_dir: str = "outputs/blogs") -> str:
        """Save generated blog post to file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create filename with date and slug
        date_str = datetime.now().strftime('%Y-%m-%d')
        filename = f"{date_str}-{blog_result['slug']}.md"
        file_path = output_path / filename

        # Format source episodes for metadata
        source_episodes = blog_result['metadata']['source_episodes']
        episode_count = blog_result['metadata']['episode_count']

        # Create source list for YAML frontmatter
        source_list = []
        for ep in source_episodes:
            source_list.append(f"  - episode: \"{ep['episode']}\"\n    podcast: \"{ep['podcast']}\"")

        sources_yaml = "\n".join(source_list)

        # Generate markdown content
        content = f"""---
title: "{blog_result['topic']}"
date: {blog_result['metadata']['generated_at']}
episode_count: {episode_count}
source_episodes:
{sources_yaml}
final_grade: {blog_result['final_grade']}
final_score: {blog_result['final_score']}
model: {blog_result['metadata']['model_used']}
inspired_by: "Tomasz Tunguz's AP English grading system"
---

# {blog_result['topic']}

{blog_result['final_post']}

---

## Generation Notes

- **Final Grade**: {blog_result['final_grade']} ({blog_result['final_score']}/100)
- **Iterations**: {len(blog_result['iterations'])}
- **Source Episodes**: {episode_count}
- **Generated**: {blog_result['metadata']['generated_at']}

### Source Episodes

{chr(10).join(f"- **{ep['podcast']}**: {ep['episode']}" for ep in source_episodes)}

### Grading History
"""
        
        # Add iteration details
        for iteration in blog_result['iterations']:
            content += f"""
**Iteration {iteration['iteration']}**: {iteration['grade']} ({iteration['score']}/100)
{iteration['feedback'][:200]}...

"""
        
        # Write to file with UTF-8 encoding (Windows compatibility)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(file_path)

    def generate_social_posts(self, blog_result: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate social media posts from blog content (Tunguz's feature)."""
        
        blog_post = blog_result['final_post']
        topic = blog_result['topic']
        
        # Extract key quotes and insights
        quotes = []
        insights = []
        
        # Simple extraction (could be enhanced with better parsing)
        sentences = blog_post.split('. ')
        for sentence in sentences:
            if len(sentence) > 50 and len(sentence) < 280:  # Twitter length
                if any(word in sentence.lower() for word in ['key', 'important', 'crucial', 'insight']):
                    insights.append(sentence.strip() + '.')
                elif '"' in sentence:
                    quotes.append(sentence.strip())
        
        # Generate Twitter posts
        twitter_prompt = f"""Generate 3 engaging Twitter posts based on this blog post about {topic}.
        
        Blog Post:
        {blog_post}
        
        Requirements:
        - Each post under 280 characters
        - Include relevant hashtags
        - Make them engaging and actionable
        - Reference key insights or quotes when possible
        
        Format as:
        POST 1: [content]
        POST 2: [content]  
        POST 3: [content]
        """
        
        # Generate LinkedIn posts
        linkedin_prompt = f"""Generate 2 LinkedIn posts based on this blog post about {topic}.
        
        Blog Post:
        {blog_post}
        
        Requirements:
        - Professional tone suitable for business audience
        - 100-200 words each
        - Include call-to-action
        - Reference source material appropriately
        
        Format as:
        POST 1: [content]
        POST 2: [content]
        """
        
        twitter_response = self._generate_with_llm(twitter_prompt)
        linkedin_response = self._generate_with_llm(linkedin_prompt)
        
        # Parse responses (simple parsing - could be enhanced)
        twitter_posts = re.findall(r'POST \d+: (.+?)(?=POST \d+:|$)', twitter_response, re.DOTALL)
        linkedin_posts = re.findall(r'POST \d+: (.+?)(?=POST \d+:|$)', linkedin_response, re.DOTALL)
        
        return {
            'twitter': [post.strip() for post in twitter_posts],
            'linkedin': [post.strip() for post in linkedin_posts],
            'quotes': quotes[:3],  # Top 3 quotable excerpts
            'insights': insights[:5]  # Top 5 key insights
        }

