# Maintainers Guide

## Making a Release

We try to automate the release process as much as possible.
GitHub Actions workflow handles publishing new releases to PyPI and updating the documentation.
The version number is set automatically using [setuptools_scm](https://pypi.org/project/setuptools-scm/)
based information obtained from git.
There are a few steps that still must be done manually, though.

### Updating the Changelog

The [Release Drafter](https://github.com/release-drafter/release-drafter) GitHub Action
will automatically keep a draft changelog at
<https://github.com/GenericMappingTools/pygmt/releases>, adding a new entry
every time a pull request (with a proper label) is merged into the main branch.
This release drafter tool has two configuration files, one for the GitHub Action
at `.github/workflows/release-drafter.yml`, and one for the changelog template
at `.github/release-drafter.yml`. Configuration settings can be found at
<https://github.com/release-drafter/release-drafter>.

The drafted release notes are not perfect, so we will need to tidy it prior to
publishing the actual release notes at [](changes.md).

1. Go to <https://github.com/GenericMappingTools/pygmt/releases> and click on the
   'Edit' button next to the current draft release note. Copy the text of the
   automatically drafted release notes under the 'Write' tab to
   `doc/changes.md`. Add a section separator `---` between the new and old
   changelog sections.
2. Update the DOI badge in the changelog. Remember to replace the DOI number
   inside the badge url.

    ```
    [![Digital Object Identifier for PyGMT vX.Y.Z](https://zenodo.org/badge/DOI/10.5281/zenodo.<INSERT-DOI-HERE>.svg)](https://doi.org/10.5281/zenodo.<INSERT-DOI-HERE>)
    ```
3. Open a new pull request using the title 'Changelog entry for vX.Y.Z' with
   the updated release notes, so that other people can help to review and
   collaborate on the changelog curation process described next.
4. Edit the change list to remove any trivial changes (updates to the README,
   typo fixes, CI configuration, test updates due to GMT releases, etc).
5. Sort the items within each section (i.e., New Features, Enhancements, etc.)
   such that similar items are located near each other (e.g., new wrapped
   modules and methods, gallery examples, API docs changes) and entries within each group
   are alphabetical.
6. Move a few important items from the main sections to the Highlights section.
7. Edit the list of people who contributed to the release, linking to their
   GitHub accounts. Sort their names by the number of commits made since the
   last release (e.g., use `git shortlog HEAD...v0.4.0 -sne`).
8. Update `doc/minversions.md` with new information on the new release version,
   including a vX.Y.Z documentation link, and minimum required versions of GMT, Python
   and core package dependencies (NumPy, pandas, Xarray). Follow
   [SPEC 0](https://scientific-python.org/specs/spec-0000/) for updates.
9. Refresh citation information. Specifically, the BibTeX in `README.md` and
   `CITATION.cff` needs to be updated with any metadata changes, including the
   DOI, release date, and version information. Please also follow
   guidelines in `AUTHORSHIP.md` for updating the author list in the BibTeX.
   More information about the `CITATION.cff` specification can be found at
   <https://github.com/citation-file-format/citation-file-format/blob/main/schema-guide.md>.

### Pushing to PyPI and Updating the Documentation

After the changelog is updated, making a release can be done by going to
<https://github.com/GenericMappingTools/pygmt/releases>, editing the draft release,
and clicking on publish. A git tag will also be created, make sure that this
tag is a proper version number (following [Semantic Versioning](https://semver.org/))
with a leading `v` (e.g., `v0.2.1`).

Once the release/tag is created, this should trigger GitHub Actions to do all the work for us.
A new source distribution will be uploaded to PyPI, a new folder with the documentation
HTML will be pushed to *gh-pages*, and the `latest` link will be updated to point to
this new folder.

### Archiving on Zenodo

Grab both the source code and baseline images ZIP files from the GitHub release page
and upload them to Zenodo using the previously reserved DOI.

### Updating the Conda Package

When a new version is released on PyPI, conda-forge's bot automatically creates version
updates for the feedstock. In most cases, the maintainers can simply merge that PR.

If changes need to be done manually, you can:

1. Fork the [pygmt feedstock repository](https://github.com/conda-forge/pygmt-feedstock) if
   you haven't already. If you have a fork, update it.
2. Update the version number and sha256 hash on `recipe/meta.yaml`. You can get the hash
   from the [PyPI "Download files" section](https://pypi.org/project/pygmt/#files).
3. Add or remove any new dependencies (most are probably only `run` dependencies).
4. Make sure the minimum support versions of all dependencies are correctly pinned.
5. Make a new branch, commit, and push the changes **to your personal fork**.
6. Create a PR against the original feedstock main.
7. Once the CI tests pass, merge the PR or ask a maintainer to do so.
