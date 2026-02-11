"""
Multi-Provider Search Tools - Centralized search tool loading and management

This module provides a unified interface for loading all available search tools from
various providers. It handles missing packages and API keys gracefully, allowing the
system to run with any combination of available tools.

Supported Search Providers (9 total):
- Exa Search (semantic/neural search)
- Tavily Search (AI-optimized search)
- Arxiv (academic papers, no API key needed)
- AskNews (news with 300k+ articles/day)
- Brave Search (privacy-focused)
- DuckDuckGo (privacy-focused, no API key needed)
- Google Serper (Google search API)
- SearXNG (metasearch aggregator, no API key needed)
- You.com (AI-powered search)
"""

import os
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool

# ==================== Tool Availability Flags ====================

EXA_AVAILABLE = False
TAVILY_AVAILABLE = False
ARXIV_AVAILABLE = False
ASKNEWS_AVAILABLE = False
BRAVE_AVAILABLE = False
DUCKDUCKGO_AVAILABLE = False
GOOGLE_SERPER_AVAILABLE = False
SEARXNG_AVAILABLE = False
YOU_AVAILABLE = False

# ==================== Import Tools with Error Handling ====================

# Exa Search (semantic/neural search)
try:
    from langchain_exa import ExaSearchResults
    EXA_AVAILABLE = True
except ImportError:
    pass

# Tavily Search (AI-optimized)
try:
    from langchain_tavily import TavilySearch
    TAVILY_AVAILABLE = True
except ImportError:
    pass

# Arxiv (academic papers)
try:
    from langchain_community.retrievers import ArxivRetriever
    ARXIV_AVAILABLE = True
except ImportError:
    pass

# AskNews (news search)
try:
    from langchain_community.tools.asknews import AskNewsSearch
    ASKNEWS_AVAILABLE = True
except ImportError:
    pass

# Brave Search
try:
    from langchain_community.tools import BraveSearch
    BRAVE_AVAILABLE = True
except ImportError:
    pass

# DuckDuckGo (no API key needed!)
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    DUCKDUCKGO_AVAILABLE = True
except ImportError:
    pass

# Google Serper
try:
    from langchain_community.utilities import GoogleSerperAPIWrapper
    GOOGLE_SERPER_AVAILABLE = True
except ImportError:
    pass

# SearXNG (metasearch)
try:
    from langchain_community.utilities import SearxSearchWrapper
    SEARXNG_AVAILABLE = True
except ImportError:
    pass

# You.com
try:
    from langchain_community.tools.you import YouSearchTool
    YOU_AVAILABLE = True
except ImportError:
    pass


# ==================== Tool Loading Functions ====================

def get_exa_tool() -> Optional[Any]:
    """Get Exa search tool if available.

    Exa provides neural/semantic search capabilities.
    Requires: EXA_API_KEY environment variable.
    """
    if not EXA_AVAILABLE:
        return None
    if not os.environ.get("EXA_API_KEY"):
        return None

    try:
        return ExaSearchResults(
            exa_api_key=os.environ["EXA_API_KEY"],
            max_results=5,
        )
    except Exception as e:
        print(f"  ⚠️  Error loading Exa: {e}")
        return None


def get_tavily_tool() -> Optional[Any]:
    """Get Tavily search tool if available.

    Tavily provides AI-optimized search for LLM applications.
    Requires: TAVILY_API_KEY environment variable.
    """
    if not TAVILY_AVAILABLE:
        return None
    if not os.environ.get("TAVILY_API_KEY"):
        return None

    try:
        return TavilySearch(max_results=5)
    except Exception as e:
        print(f"  ⚠️  Error loading Tavily: {e}")
        return None


def get_arxiv_tool() -> Optional[Any]:
    """Get Arxiv academic paper search tool if available.

    Arxiv provides access to academic papers across multiple disciplines.
    No API key needed!
    """
    if not ARXIV_AVAILABLE:
        return None

    try:
        # Wrap ArxivRetriever as a tool
        retriever = ArxivRetriever(load_max_docs=3)

        @tool
        def arxiv_search(query: str) -> str:
            """Search for academic papers on arXiv.

            Use this to find research papers, preprints, and academic publications.
            Good for scientific, mathematical, and technical topics.

            Args:
                query: Search query for academic papers (e.g., "attention mechanisms in transformers")

            Returns:
                Formatted search results with titles, authors, and summaries.
            """
            try:
                docs = retriever.get_relevant_documents(query)
                if not docs:
                    return "No papers found."

                results = []
                for i, doc in enumerate(docs[:3], 1):
                    title = doc.metadata.get('Title', 'Unknown')
                    authors = doc.metadata.get('Authors', 'Unknown')
                    summary = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    results.append(f"{i}. {title}\n   Authors: {authors}\n   Summary: {summary}")

                return "\n\n".join(results)
            except Exception as e:
                return f"Error searching arXiv: {e}"

        return arxiv_search
    except Exception as e:
        print(f"  ⚠️  Error loading Arxiv: {e}")
        return None


def get_asknews_tool() -> Optional[Any]:
    """Get AskNews tool if available.

    AskNews provides access to 300k+ news articles per day with AI-powered search.
    Requires: ASKNEWS_CLIENT_ID and ASKNEWS_CLIENT_SECRET environment variables.
    """
    if not ASKNEWS_AVAILABLE:
        return None
    if not os.environ.get("ASKNEWS_CLIENT_ID") or not os.environ.get("ASKNEWS_CLIENT_SECRET"):
        return None

    try:
        return AskNewsSearch()
    except Exception as e:
        print(f"  ⚠️  Error loading AskNews: {e}")
        return None


def get_brave_tool() -> Optional[Any]:
    """Get Brave search tool if available.

    Brave provides privacy-focused web search.
    Requires: BRAVE_SEARCH_API_KEY environment variable.
    """
    if not BRAVE_AVAILABLE:
        return None
    if not os.environ.get("BRAVE_SEARCH_API_KEY"):
        return None

    try:
        return BraveSearch.from_api_key(
            api_key=os.environ["BRAVE_SEARCH_API_KEY"],
            search_kwargs={"count": 5}
        )
    except Exception as e:
        print(f"  ⚠️  Error loading Brave: {e}")
        return None


def get_duckduckgo_tool() -> Optional[Any]:
    """Get DuckDuckGo search tool if available.

    DuckDuckGo provides privacy-focused web search.
    No API key needed!
    """
    if not DUCKDUCKGO_AVAILABLE:
        return None

    try:
        return DuckDuckGoSearchRun()
    except Exception as e:
        print(f"  ⚠️  Error loading DuckDuckGo: {e}")
        return None


def get_google_serper_tool() -> Optional[Any]:
    """Get Google Serper search tool if available.

    Google Serper provides access to Google Search results via API.
    Requires: SERPER_API_KEY environment variable.
    """
    if not GOOGLE_SERPER_AVAILABLE:
        return None
    if not os.environ.get("SERPER_API_KEY"):
        return None

    try:
        search = GoogleSerperAPIWrapper()

        @tool
        def google_serper_search(query: str) -> str:
            """Search Google using Serper API.

            Use this for general web searches to find current information, facts,
            news, and websites.

            Args:
                query: Search query (e.g., "current weather in San Francisco")

            Returns:
                Search results from Google.
            """
            return search.run(query)

        return google_serper_search
    except Exception as e:
        print(f"  ⚠️  Error loading Google Serper: {e}")
        return None


def get_searxng_tool() -> Optional[Any]:
    """Get SearXNG metasearch tool if available.

    SearXNG aggregates results from multiple search engines.
    No API key needed! Can optionally configure SEARXNG_URL for custom instance.
    """
    if not SEARXNG_AVAILABLE:
        return None

    # SearXNG requires a URL to a SearXNG instance
    searxng_url = os.environ.get("SEARXNG_URL", "https://searx.be")

    try:
        search = SearxSearchWrapper(searx_host=searxng_url)

        @tool
        def searxng_search(query: str) -> str:
            """Search using SearXNG metasearch engine.

            SearXNG aggregates results from multiple search engines (Google, Bing,
            DuckDuckGo, etc.) to provide comprehensive results.

            Args:
                query: Search query

            Returns:
                Aggregated search results from multiple engines.
            """
            try:
                return search.run(query)
            except Exception as e:
                return f"Error searching SearXNG: {e}"

        return searxng_search
    except Exception as e:
        print(f"  ⚠️  Error loading SearXNG: {e}")
        return None


def get_you_tool() -> Optional[Any]:
    """Get You.com search tool if available.

    You.com provides AI-powered search capabilities.
    Requires: YDC_API_KEY environment variable.
    """
    if not YOU_AVAILABLE:
        return None
    if not os.environ.get("YDC_API_KEY"):
        return None

    try:
        return YouSearchTool(api_key=os.environ["YDC_API_KEY"])
    except Exception as e:
        print(f"  ⚠️  Error loading You.com: {e}")
        return None


# ==================== Main Function ====================

def get_all_search_tools(verbose: bool = True) -> List[Any]:
    """
    Get all available search tools based on installed packages and API keys.

    This function attempts to load all 9 supported search providers and returns
    a list of successfully loaded tools. It gracefully handles missing packages
    and API keys.

    Args:
        verbose: If True, print information about tool availability

    Returns:
        List of available search tool instances ready to use with LangChain agents.
    """
    tools = []

    if verbose:
        print("\n📦 Loading Search Tools...")
        print("="*60)

    # Track loaded and missing tools
    loaded_tools = []
    missing_tools = []

    # Define all search tool loaders with metadata
    tool_loaders = [
        ("Exa Search", get_exa_tool, "EXA_API_KEY"),
        ("Tavily Search", get_tavily_tool, "TAVILY_API_KEY"),
        ("Arxiv (Academic)", get_arxiv_tool, None),  # No key needed
        ("AskNews", get_asknews_tool, "ASKNEWS_CLIENT_ID + SECRET"),
        ("Brave Search", get_brave_tool, "BRAVE_SEARCH_API_KEY"),
        ("DuckDuckGo", get_duckduckgo_tool, None),  # No key needed
        ("Google Serper", get_google_serper_tool, "SERPER_API_KEY"),
        ("SearXNG", get_searxng_tool, "SEARXNG_URL (optional)"),
        ("You.com", get_you_tool, "YDC_API_KEY"),
    ]

    # Try to load each search tool
    for name, loader, key_name in tool_loaders:
        tool = loader()
        if tool:
            tools.append(tool)
            loaded_tools.append(name)
            if verbose:
                print(f"✅ {name}")
        else:
            missing_tools.append((name, key_name))
            if verbose:
                if key_name:
                    print(f"⚠️  {name} (missing {key_name})")
                else:
                    print(f"⚠️  {name} (package not installed)")

    if verbose:
        print("="*60)
        print(f"✅ Loaded {len(loaded_tools)} search tools")
        print(f"⚠️  {len(missing_tools)} tools unavailable")
        print(f"\nTotal available: {len(tools)} search tools")

        if len(tools) == 0:
            print("\n⚠️  WARNING: No search tools available!")
            print("Install at least one search tool package:")
            print("  pip install langchain-exa langchain-tavily")
            print("  pip install duckduckgo-search  # No API key needed!")
            print("  pip install arxiv pymupdf  # No API key needed!")

    return tools


def get_search_tool_info() -> Dict[str, Any]:
    """Get detailed information about search tool availability.

    Returns:
        Dictionary containing:
        - search_tools: Dict mapping tool names to availability (bool)
        - packages_available: Dict mapping tool names to package installation status
        - loaded_count: Number of tools that can be loaded
    """
    # Check which tools can actually be loaded (package + API key)
    search_tools_ready = {
        "exa": EXA_AVAILABLE and bool(os.environ.get("EXA_API_KEY")),
        "tavily": TAVILY_AVAILABLE and bool(os.environ.get("TAVILY_API_KEY")),
        "arxiv": ARXIV_AVAILABLE,
        "asknews": ASKNEWS_AVAILABLE and bool(os.environ.get("ASKNEWS_CLIENT_ID")),
        "brave": BRAVE_AVAILABLE and bool(os.environ.get("BRAVE_SEARCH_API_KEY")),
        "duckduckgo": DUCKDUCKGO_AVAILABLE,
        "google_serper": GOOGLE_SERPER_AVAILABLE and bool(os.environ.get("SERPER_API_KEY")),
        "searxng": SEARXNG_AVAILABLE,
        "you": YOU_AVAILABLE and bool(os.environ.get("YDC_API_KEY")),
    }

    # Check package installation status
    packages_available = {
        "exa": EXA_AVAILABLE,
        "tavily": TAVILY_AVAILABLE,
        "arxiv": ARXIV_AVAILABLE,
        "asknews": ASKNEWS_AVAILABLE,
        "brave": BRAVE_AVAILABLE,
        "duckduckgo": DUCKDUCKGO_AVAILABLE,
        "google_serper": GOOGLE_SERPER_AVAILABLE,
        "searxng": SEARXNG_AVAILABLE,
        "you": YOU_AVAILABLE,
    }

    return {
        "search_tools": search_tools_ready,
        "packages_available": packages_available,
        "loaded_count": sum(search_tools_ready.values()),
        "total_possible": len(search_tools_ready),
    }


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("MULTI-PROVIDER SEARCH TOOLS TEST")
    print("="*60)

    # Get all available tools
    tools = get_all_search_tools(verbose=True)

    # Show detailed info
    print("\n" + "="*60)
    print("DETAILED TOOL INFO")
    print("="*60)

    info = get_search_tool_info()

    print(f"\n📊 Search Tools Status ({info['loaded_count']}/{info['total_possible']} ready):")
    for name, available in info["search_tools"].items():
        package_status = "📦 Installed" if info["packages_available"][name] else "❌ Not installed"
        tool_status = "✅ Ready" if available else "⚠️  Missing API key" if info["packages_available"][name] else "❌ Not installed"
        print(f"  {name:15} | {package_status:20} | {tool_status}")

    print("\n" + "="*60)
    print(f"✅ TEST COMPLETE - {len(tools)} tools ready to use")
    print("="*60)

    # Print tool names for debugging
    if tools:
        print("\n📝 Loaded tool names:")
        for tool in tools:
            print(f"  - {tool.name}")
