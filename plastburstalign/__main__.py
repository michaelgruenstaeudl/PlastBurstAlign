import sys

from .logging_ops import logger as log
from .plast_burst_align import PlastBurstAlign
from .user_parameters import UserParametersScript


def _dashboard_mode(flag: bool | None) -> str:
    if flag is True:
        return "rich"
    if flag is False:
        return "off"
    return "auto"


def _run_estimate(params: dict) -> int:
    """Extract + clean, print a pre-flight estimate, exit without aligning."""
    from .align_runtime import clear_stale_outputs
    from .autotune import autotune
    from .estimate import preflight_estimate, render_preflight
    from .extraction_ops import DataCleaning, ExtractAndCollect
    from .seqfeature_ops import PlastidData
    from .ui import detect_plain_environment

    clear_stale_outputs(str(params.get("out_dir", ".")))
    data = PlastidData(params)
    try:
        ExtractAndCollect(data, params).extract()
        DataCleaning(data, params).clean()
        mode = params.get("select_mode")
        mode_key = str(getattr(mode, "value", mode) or "cds").lower()
        prefix = "prot" if mode_key == "cds" else "nucl"
        try:
            summary = data.store.get_region_summary(seq_type=prefix)
        except Exception:
            summary = data.store.get_region_summary(seq_type="nucl")
        prot_summary = None
        if mode_key == "cds":
            try:
                prot_summary = data.store.get_region_summary(seq_type="prot")
            except Exception:
                prot_summary = None
        threads = params.get("num_threads_requested", params.get("num_threads", "auto"))
        plan = autotune(
            num_threads="auto" if threads is None else threads,
            smt=params.get("smt"),
            memory_limit=params.get("memory_limit"),
        )
        want_sub = str(params.get("subdivide", "auto")).lower() != "off"
        est = preflight_estimate(
            summary if isinstance(summary, dict) else {},
            mode=mode_key,
            ntax=int(getattr(data, "num_taxa", 0) or 0) or None,
            plan=plan,
            subdivide=want_sub,
            prot_summary=prot_summary if isinstance(prot_summary, dict) else None,
            keep_unaligned=bool(params.get("keep_unaligned", True)),
        )
        if detect_plain_environment() is None:
            try:
                render_preflight(est)
                return 0
            except Exception:
                pass
        print(est.to_text())
        return 0
    finally:
        try:
            data.store.close()
        except Exception:
            pass


def _print_scaling_report(burst_align: PlastBurstAlign, dashboard: bool | None) -> None:
    aligncoord = burst_align.last_aligncoord
    if aligncoord is None:
        return
    report = getattr(aligncoord, "_scheduler_report", None)
    if report is None:
        return
    from .scaling import ScalingReport
    from .ui import format_run_summary_text, render_run_summary

    cores = int(getattr(report, "total_cores", 0) or 0) or 1
    plan = getattr(aligncoord, "_plan", None)
    if plan is not None:
        cores = max(cores, int(getattr(plan, "cpu_budget", cores) or cores))
    scaling = ScalingReport.from_scheduler(
        report,
        cores=cores,
        supermatrix_summary=getattr(aligncoord, "_supermatrix_info", None),
        scratch_stats=aligncoord.scratch_stats(),
    )
    if dashboard is False:
        print(format_run_summary_text(scaling))
    else:
        try:
            render_run_summary(None, scaling)
        except Exception:
            print(format_run_summary_text(scaling))


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--version" in args or "-V" in args:
        from ._meta import __version__

        print(__version__)
        return 0
    if "--doctor" in args:
        from .doctor import main as doctor_main

        cleaned = [a for a in args if a != "--doctor"]
        return int(doctor_main(cleaned))

    # Rebuild argv for argparse when called with an explicit list.
    if argv is not None:
        old = sys.argv
        try:
            sys.argv = [old[0], *argv]
            params = UserParametersScript()
        finally:
            sys.argv = old
    else:
        params = UserParametersScript()

    if params.get("time_budget") is not None:
        log.warning(
            "--time-budget is opt-in and non-reproducible; "
            "host-adaptive scheduling is not fully wired yet (value recorded only)"
        )

    if params.get("estimate"):
        return _run_estimate(params)

    from .ui import close_sink, make_sink

    dashboard = params.get("dashboard")
    sink = make_sink(
        _dashboard_mode(dashboard if isinstance(dashboard, bool) else None)
    )
    params["progress_sink"] = sink
    burst_align = PlastBurstAlign(params)
    try:
        burst_align.execute()
        _print_scaling_report(
            burst_align, dashboard if isinstance(dashboard, bool) else None
        )
    finally:
        close_sink(sink)
    print("\nend of script\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
