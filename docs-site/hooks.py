"""MkDocs hooks for the tiervec documentation.

The Markdown files of this repository are written to be read on GitHub, where a link such as
`docs/STATUS.md` or `../CHANGELOG.md` is a path in the repository. The site is built from the
`docs/` folder only, so three things need help, and this file does all of them at build time.
The source files stay as they are, so there is one source for GitHub and for the site.

1. Pages from the repository root. The README is the home page, and CHANGELOG, CONTRIBUTING,
   SECURITY, SUPPORT and THIRD_PARTY are pages too. They are added to the site as generated
   files with the same text.
2. Links. Every relative link is resolved against the path of the file it is written in. A link
   to a file that is a page of the site becomes a link to that page. A link to any other file or
   folder of the repository (a source file, a workflow, the license) becomes an absolute GitHub
   URL pinned to `main`. A link that points nowhere is a warning, and `mkdocs build --strict`
   fails on a warning.
3. The banner. Every page starts, after its title, with a note that the library is experimental
   and unreleased.
"""

import logging
import os
import posixpath
import re

from mkdocs.structure.files import File

log = logging.getLogger("mkdocs.hooks.tiervec")

REPO_URL = "https://github.com/humemai/tiervec"
BLOB_URL = REPO_URL + "/blob/main/"
TREE_URL = REPO_URL + "/tree/main/"
EDIT_URL = REPO_URL + "/edit/main/"

# Files of the repository root that are pages of the site: repository path -> path in the site.
ROOT_PAGES = {
    "README.md": "index.md",
    "CHANGELOG.md": "CHANGELOG.md",
    "CONTRIBUTING.md": "CONTRIBUTING.md",
    "SECURITY.md": "SECURITY.md",
    "SUPPORT.md": "SUPPORT.md",
    "THIRD_PARTY.md": "THIRD_PARTY.md",
}

BANNER = (
    '<div class="hm-banner" markdown>\n'
    "**Experimental and unreleased.** The API is not stable and nothing here is wired into a "
    "database engine yet. [What may change](API.md#stability-policy-for-0x).\n"
    "</div>\n"
)

SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
INLINE_LINK = re.compile(r'(\]\()\s*(<[^>]+>|[^)\s]+)((?:\s+"[^"]*")?\s*\))')
REFERENCE_LINK = re.compile(r"^(\s{0,3}\[[^\]]+\]:\s*)(\S+)")
FENCE = re.compile(r"^\s*(```|~~~)")

# Set by on_files: the root of the repository, and the site path of every page by repository path.
_state = {"root": "", "pages": {}}


def _repo_path(src_uri):
    """The path in the repository of the file that a page of the site is written from."""
    for repo_path, uri in ROOT_PAGES.items():
        if uri == src_uri:
            return repo_path
    return "docs/" + src_uri


def on_files(files, config):
    root = os.path.dirname(os.path.abspath(config.config_file_path))
    _state["root"] = root
    pages = {}
    for repo_path, uri in ROOT_PAGES.items():
        with open(os.path.join(root, repo_path), encoding="utf-8") as handle:
            text = handle.read()
        files.append(File.generated(config, uri, content=text))
        pages[repo_path] = uri
    for page in files.documentation_pages():
        if page.src_uri not in ROOT_PAGES.values() and not page.src_uri.startswith("brand/"):
            pages["docs/" + page.src_uri] = page.src_uri
    _state["pages"] = pages
    return files


def on_pre_page(page, config, files):
    page.edit_url = EDIT_URL + _repo_path(page.file.src_uri)
    return page


def _rewrite(target, repo_path, src_uri):
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    if SCHEME.match(target) or target.startswith(("#", "/")):
        return target
    path, hash_sign, fragment = target.partition("#")
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(repo_path), path))
    if resolved == ".." or resolved.startswith("../"):
        log.warning("%s: the link %r leaves the repository", repo_path, target)
        return target
    page_uri = _state["pages"].get(resolved)
    if page_uri is not None:
        return posixpath.relpath(page_uri, posixpath.dirname(src_uri) or ".") + hash_sign + fragment
    full = os.path.join(_state["root"], resolved)
    if os.path.isdir(full):
        return TREE_URL + resolved + hash_sign + fragment
    if os.path.isfile(full):
        return BLOB_URL + resolved + hash_sign + fragment
    log.warning("%s: the link %r points to %r, which does not exist", repo_path, target, resolved)
    return target


def on_page_markdown(markdown, page, config, files):
    src_uri = page.file.src_uri
    repo_path = _repo_path(src_uri)
    lines = markdown.split("\n")
    in_fence = False
    for index, line in enumerate(lines):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        line = INLINE_LINK.sub(
            lambda m: m.group(1) + _rewrite(m.group(2), repo_path, src_uri) + m.group(3), line
        )
        line = REFERENCE_LINK.sub(
            lambda m: m.group(1) + _rewrite(m.group(2), repo_path, src_uri), line
        )
        lines[index] = line
    # The banner goes after the first heading, or at the top if the page has none first.
    for index, line in enumerate(lines):
        if line.strip():
            at = index + 1 if line.startswith("# ") else index
            lines[at:at] = ["", *BANNER.rstrip("\n").split("\n"), ""]
            break
    return "\n".join(lines)
