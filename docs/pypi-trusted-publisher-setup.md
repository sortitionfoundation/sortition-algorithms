# PyPI Trusted Publisher Setup

This page is the one-time configuration runbook for switching the
`sortition-algorithms` PyPI release pipeline from a long-lived API token
(`PYPI_TOKEN` repo secret) to PyPI's
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) mechanism.

The workflow change itself (in `.github/workflows/on-release-main.yml`) has
already been made: the publish job requests an OIDC token (`id-token: write`),
runs in a GitHub `pypi` environment, and uses
[`pypa/gh-action-pypi-publish`](https://github.com/marketplace/actions/pypi-publish)
to upload to PyPI. With Trusted Publishing, GitHub Actions exchanges its
short-lived OIDC token for a 15-minute PyPI API token at upload time — no
secret is stored in the repo.

What's left is the manual setup on pypi.org and on GitHub. Do these in order.

## 1. Configure the publisher on PyPI

You need to be a maintainer/owner of the `sortition-algorithms` project on
PyPI.

1. Sign in at <https://pypi.org/>.
2. Go to your projects: <https://pypi.org/manage/projects/>.
3. Click **Manage** next to `sortition-algorithms`.
4. In the project sidebar, click **Publishing**.
5. Under "Add a new publisher", select the **GitHub** tab and fill in:

   | Field | Value |
   | --- | --- |
   | PyPI Project Name | `sortition-algorithms` |
   | Owner | `sortitionfoundation` |
   | Repository name | `sortition-algorithms` |
   | Workflow name | `on-release-main.yml` |
   | Environment name | `pypi` |

6. Click **Add**. The publisher should appear at the top of the page.

The Workflow name is the *filename* of the workflow inside
`.github/workflows/`, not the `name:` field at the top. The Environment name
must match the `environment.name` in the publish job exactly (case-sensitive).

## 2. Configure the matching environment on GitHub

The PyPI publisher is bound to a GitHub
[deployment environment](https://docs.github.com/en/actions/deployment/targeting-different-environments/using-environments-for-deployment)
called `pypi`. This must exist in the repo or the workflow will fail.

1. Go to the repo on GitHub:
   <https://github.com/sortitionfoundation/sortition-algorithms>.
2. **Settings** → **Environments** → **New environment**.
3. Name it exactly `pypi` and click **Configure environment**.
4. Recommended hardening (all optional, but worth doing):
   - **Required reviewers**: add yourself and any other release approvers.
     Each release will then pause until a reviewer approves the publish job —
     a useful sanity check, since release tags can be created accidentally.
   - **Deployment branches and tags**: restrict to tags only. A pattern like
     `v*` or `*.*.*` matches our release tags and prevents the environment
     from being targeted by ordinary branch pushes.
5. Do **not** add any environment secrets — the whole point is that there's
   no secret to store.

## 3. Cut over

1. Merge the workflow change to `main` if you haven't already.
2. Cut a real release on GitHub (Releases → Draft a new release → publish).
   The first release after the cutover is the live test.
3. Watch the `release-main` workflow run. The `publish` job will pause for
   environment approval if you configured required reviewers. Approve it.
4. Confirm the new version appears on
   <https://pypi.org/project/sortition-algorithms/>.

## 4. Remove the old API token

Only do this **after** a successful release via Trusted Publishing.

1. **Revoke the token on PyPI** — go to
   <https://pypi.org/manage/account/token/>, find the token used by the old
   workflow, and delete it.
2. **Delete the GitHub secret** — repo **Settings** → **Secrets and
   variables** → **Actions** → delete `PYPI_TOKEN`.

Removing the GitHub secret without revoking the PyPI token leaves a valid
token sitting on PyPI; revoking on PyPI without removing the GitHub secret
leaves a misleading dead secret in the repo. Do both.

## Troubleshooting

These errors come from PyPI's `mint-token` endpoint, which the action calls
to exchange the OIDC token for a short-lived API token. The full reference
is in the
[PyPI troubleshooting docs](https://docs.pypi.org/trusted-publishers/troubleshooting/).

- **`invalid-publisher`** — the OIDC token doesn't match any publisher
  configured for the project. Almost always a typo or mismatch:
  - Workflow filename on PyPI ≠ actual filename in `.github/workflows/`.
  - Environment name on PyPI ≠ `environment.name` in the workflow.
  - Repo owner / repo name changed (e.g. transferred or renamed) without
    updating the publisher on PyPI.
- **`invalid-pending-publisher`** — same idea, but for a pending publisher
  used to create a brand-new project. Doesn't apply here since the project
  already exists.
- **Job fails with "missing id-token permission"** — the `permissions` block
  on the publish job is missing `id-token: write`. Without it GitHub refuses
  to issue an OIDC token at all.
- **Job hangs or skips** — check the **Environments** tab in the repo. If
  required reviewers are configured, the publish job waits for approval.
- **Workflow file renamed** — if `on-release-main.yml` is ever renamed,
  update the publisher config on PyPI to match before the next release, or
  add a second publisher for the new filename.

## Reference

- [PyPA `pypi-publish` action](https://github.com/marketplace/actions/pypi-publish)
- [PyPI Trusted Publishers — overview](https://docs.pypi.org/trusted-publishers/)
- [Adding a publisher to an existing project](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
- [Using a publisher (workflow examples)](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
- [Troubleshooting](https://docs.pypi.org/trusted-publishers/troubleshooting/)
