#!/usr/bin/env python3
"""
Test Multi-Provider Search Tools

Tests that all search tools load correctly and provides a summary
of which tools are available based on installed packages and API keys.

Usage:
    python test_multi_search.py
"""

from dotenv import load_dotenv
load_dotenv()

from rich import print as rprint
from spec_tool_call.tools.multi_search_tool import (
    get_all_search_tools,
    get_search_tool_info,
)


def test_search_tools():
    """Test loading and availability of search tools."""
    rprint("\n" + "="*80)
    rprint("[bold cyan]MULTI-PROVIDER SEARCH TOOLS TEST[/bold cyan]")
    rprint("="*80)

    # Load all available tools
    rprint("\n📦 Loading search tools...")
    tools = get_all_search_tools(verbose=True)

    # Get detailed info
    info = get_search_tool_info()

    # Print summary
    rprint("\n" + "="*80)
    rprint("DETAILED AVAILABILITY")
    rprint("="*80)

    search_tools_status = info["search_tools"]
    packages_status = info["packages_available"]

    rprint(f"\n📊 Tool Status ({info['loaded_count']}/{info['total_possible']} ready):\n")

    tools_info = [
        ("Exa Search", "exa", "EXA_API_KEY"),
        ("Tavily Search", "tavily", "TAVILY_API_KEY"),
        ("Arxiv (Academic)", "arxiv", None),
        ("AskNews", "asknews", "ASKNEWS_CLIENT_ID + SECRET"),
        ("Brave Search", "brave", "BRAVE_SEARCH_API_KEY"),
        ("DuckDuckGo", "duckduckgo", None),
        ("Google Serper", "google_serper", "SERPER_API_KEY"),
        ("SearXNG", "searxng", "SEARXNG_URL (optional)"),
        ("You.com", "you", "YDC_API_KEY"),
    ]

    for name, key, api_key in tools_info:
        package_installed = packages_status.get(key, False)
        tool_ready = search_tools_status.get(key, False)

        if tool_ready:
            status = "✅ READY"
            color = "green"
        elif package_installed:
            status = f"⚠️  MISSING API KEY ({api_key})" if api_key else "⚠️  NOT CONFIGURED"
            color = "yellow"
        else:
            status = "❌ PACKAGE NOT INSTALLED"
            color = "red"

        rprint(f"  [{color}]{name:25} {status}[/{color}]")

    # Print installation instructions if needed
    missing_packages = [
        key for key, installed in packages_status.items()
        if not installed
    ]

    if missing_packages:
        rprint("\n📝 To install missing packages:")
        install_cmds = {
            "exa": "pip install langchain-exa",
            "tavily": "pip install langchain-tavily",
            "arxiv": "pip install arxiv pymupdf",
            "asknews": "pip install asknews",
            "brave": "pip install langchain-community",
            "duckduckgo": "pip install duckduckgo-search",
            "google_serper": "pip install langchain-community",
            "searxng": "pip install langchain-community",
            "you": "pip install langchain-community",
        }

        for pkg in missing_packages:
            if pkg in install_cmds:
                rprint(f"  {install_cmds[pkg]}")

    # Print tool names for debugging
    if tools:
        rprint("\n🔧 Loaded Tool Names (for LangChain):")
        for tool in tools:
            rprint(f"  - {tool.name}")

    # Recommendations
    rprint("\n💡 Recommendations:")
    if info['loaded_count'] == 0:
        rprint("  [red]⚠️  No search tools available! Install at least one:[/red]")
        rprint("  [yellow]  Quick start (no API key): pip install duckduckgo-search arxiv pymupdf[/yellow]")
    elif info['loaded_count'] < 3:
        rprint("  [yellow]⚠️  Few search tools available. Consider adding more for redundancy.[/yellow]")
    else:
        rprint("  [green]✅ Good coverage! Multiple search tools available.[/green]")

    rprint("\n" + "="*80)
    rprint(f"[bold green]TEST COMPLETE - {len(tools)} tools ready to use[/bold green]")
    rprint("="*80)


if __name__ == "__main__":
    test_search_tools()
