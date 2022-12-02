import nox

nox.options.sessions = ["tests", "docs", "serve"]


@nox.session
def tests(session: nox.Session) -> None:
    """
    Run the unit and regular tests. a
    """
    session.install(".[test]")
    session.run("pytest", *session.posargs)
