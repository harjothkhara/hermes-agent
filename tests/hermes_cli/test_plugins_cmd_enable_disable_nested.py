"""Tests for nested/alias-normalized enable & disable flows.

Companion to test_plugins_cmd_category_discovery.py. That file covers the
*listing* side of nested category plugins (issue #41066). These tests cover
the *mutation* side: `hermes plugins enable/disable` must resolve a bare name
OR a full path-derived key (e.g. `observability/nemo_relay`) to the canonical
registry key and write THAT — the same string PluginManager gates on — so a
nested bundled plugin can actually be toggled.
"""

import sys  # noqa: F401
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _make_plugin_dir(parent: Path, name: str, manifest: dict) -> Path:
    d = parent / name
    d.mkdir(parents=True, exist_ok=True)
    import yaml
    (d / "plugin.yaml").write_text(yaml.dump(manifest), encoding="utf-8")
    (d / "__init__.py").write_text("def register(ctx): pass\n", encoding="utf-8")
    return d


def _make_category_plugin(parent: Path, category: str, name: str, manifest: dict) -> Path:
    return _make_plugin_dir(parent / category, name, manifest)


@pytest.fixture
def nested_plugin_env(tmp_path):
    """A user-plugins dir containing one nested and one flat plugin, with the
    bundled dir pointed at an empty path. Returns the tmp_path."""
    _make_category_plugin(tmp_path, "observability", "nemo_relay", {
        "name": "nemo_relay", "version": "1.0.0", "description": "relay obs"
    })
    _make_plugin_dir(tmp_path, "disk-cleanup", {
        "name": "disk-cleanup", "version": "1.0.0"
    })
    return tmp_path


# ---------------------------------------------------------------------------
# _resolve_plugin_key
# ---------------------------------------------------------------------------


class TestResolvePluginKey:
    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_full_key_resolves_to_itself(self, mock_user, mock_bundled, nested_plugin_env):
        from hermes_cli.plugins_cmd import _resolve_plugin_key
        mock_user.return_value = nested_plugin_env
        mock_bundled.return_value = nested_plugin_env / "nonexistent"
        assert _resolve_plugin_key("observability/nemo_relay") == "observability/nemo_relay"

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_bare_leaf_name_resolves_to_key(self, mock_user, mock_bundled, nested_plugin_env):
        from hermes_cli.plugins_cmd import _resolve_plugin_key
        mock_user.return_value = nested_plugin_env
        mock_bundled.return_value = nested_plugin_env / "nonexistent"
        # "nemo_relay" (bare) must normalize to the path-derived key.
        assert _resolve_plugin_key("nemo_relay") == "observability/nemo_relay"

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_flat_plugin_resolves_to_name(self, mock_user, mock_bundled, nested_plugin_env):
        from hermes_cli.plugins_cmd import _resolve_plugin_key
        mock_user.return_value = nested_plugin_env
        mock_bundled.return_value = nested_plugin_env / "nonexistent"
        assert _resolve_plugin_key("disk-cleanup") == "disk-cleanup"

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_unknown_returns_none(self, mock_user, mock_bundled, nested_plugin_env):
        from hermes_cli.plugins_cmd import _resolve_plugin_key
        mock_user.return_value = nested_plugin_env
        mock_bundled.return_value = nested_plugin_env / "nonexistent"
        assert _resolve_plugin_key("does-not-exist") is None

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_ambiguous_leaf_name_returns_none(self, mock_user, mock_bundled, tmp_path):
        """Same leaf name under two categories must NOT silently pick one."""
        from hermes_cli.plugins_cmd import _resolve_plugin_key
        _make_category_plugin(tmp_path, "image_gen", "openai", {"name": "image-gen-openai"})
        _make_category_plugin(tmp_path, "model-providers", "openai", {"name": "mp-openai"})
        mock_user.return_value = tmp_path
        mock_bundled.return_value = tmp_path / "nonexistent"
        # Bare "openai" is ambiguous -> None; the full key still resolves.
        assert _resolve_plugin_key("openai") is None
        assert _resolve_plugin_key("image_gen/openai") == "image_gen/openai"


# ---------------------------------------------------------------------------
# cmd_enable / cmd_disable — write the canonical key
# ---------------------------------------------------------------------------


class TestEnableDisableNested:
    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set())
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_enable_bare_name_writes_key(
        self, mock_en, mock_dis, mock_save_en, mock_save_dis,
        mock_user, mock_bundled, nested_plugin_env,
    ):
        from hermes_cli.plugins_cmd import cmd_enable
        mock_user.return_value = nested_plugin_env
        mock_bundled.return_value = nested_plugin_env / "nonexistent"

        cmd_enable("nemo_relay")  # bare name

        saved = mock_save_en.call_args[0][0]
        # The canonical key — NOT the bare name — must be persisted, because
        # that is what PluginManager matches when deciding to load.
        assert "observability/nemo_relay" in saved
        assert "nemo_relay" not in saved or "observability/nemo_relay" in saved

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set())
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_enable_full_key_writes_key(
        self, mock_en, mock_dis, mock_save_en, mock_save_dis,
        mock_user, mock_bundled, nested_plugin_env,
    ):
        from hermes_cli.plugins_cmd import cmd_enable
        mock_user.return_value = nested_plugin_env
        mock_bundled.return_value = nested_plugin_env / "nonexistent"

        cmd_enable("observability/nemo_relay")
        saved = mock_save_en.call_args[0][0]
        assert "observability/nemo_relay" in saved

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set())
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_disable_bare_name_writes_key_and_clears_alias(
        self, mock_en, mock_dis, mock_save_en, mock_save_dis,
        mock_user, mock_bundled, nested_plugin_env,
    ):
        from hermes_cli.plugins_cmd import cmd_disable
        mock_user.return_value = nested_plugin_env
        mock_bundled.return_value = nested_plugin_env / "nonexistent"
        # Simulate an existing config where the plugin was enabled under the
        # legacy bare name — disabling must clear that too, or the plugin would
        # keep loading (PluginManager accepts the bare name as well).
        mock_en.return_value = {"nemo_relay"}

        cmd_disable("nemo_relay")
        saved_dis = mock_save_dis.call_args[0][0]
        saved_en = mock_save_en.call_args[0][0]
        assert "observability/nemo_relay" in saved_dis
        assert "nemo_relay" not in saved_en  # stale bare alias dropped

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set())
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_disable_bundled_platform_refuses_ignored_disabled_row(
        self, mock_en, mock_dis, mock_save_en, mock_save_dis,
        mock_user, mock_bundled, tmp_path, capsys,
    ):
        from hermes_cli.plugins_cmd import cmd_disable

        bundled_dir = tmp_path / "bundled"
        _make_category_plugin(
            bundled_dir,
            "platforms",
            "discord",
            {
                "name": "discord-platform",
                "kind": "platform",
                "version": "1.0.0",
            },
        )
        user_dir = tmp_path / "user"
        user_dir.mkdir()
        mock_user.return_value = user_dir
        mock_bundled.return_value = bundled_dir

        with pytest.raises(SystemExit):
            cmd_disable("discord")

        out = capsys.readouterr().out
        assert "gateway.platforms.discord.enabled" in out
        mock_save_en.assert_not_called()
        mock_save_dis.assert_not_called()

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_disabled_set", return_value={"platforms/discord"})
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value={"discord-platform"})
    def test_enable_bundled_platform_cleans_legacy_disabled_alias(
        self, mock_en, mock_dis, mock_save_en, mock_save_dis,
        mock_user, mock_bundled, tmp_path, capsys,
    ):
        from hermes_cli.plugins_cmd import cmd_enable

        bundled_dir = tmp_path / "bundled"
        _make_category_plugin(
            bundled_dir,
            "platforms",
            "discord",
            {
                "name": "discord-platform",
                "kind": "platform",
                "version": "1.0.0",
            },
        )
        user_dir = tmp_path / "user"
        user_dir.mkdir()
        mock_user.return_value = user_dir
        mock_bundled.return_value = bundled_dir

        cmd_enable("discord")

        out = capsys.readouterr().out
        assert "gateway.platforms.discord.enabled" in out
        saved_disabled = mock_save_dis.call_args[0][0]
        assert "platforms/discord" not in saved_disabled
        mock_save_en.assert_not_called()

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_enable_unknown_plugin_exits(self, mock_user, mock_bundled, nested_plugin_env):
        from hermes_cli.plugins_cmd import cmd_enable
        mock_user.return_value = nested_plugin_env
        mock_bundled.return_value = nested_plugin_env / "nonexistent"
        with pytest.raises(SystemExit):
            cmd_enable("does-not-exist")

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set())
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_enable_flat_plugin_unchanged(
        self, mock_en, mock_dis, mock_save_en, mock_save_dis,
        mock_user, mock_bundled, nested_plugin_env,
    ):
        """Flat plugins keep writing their bare name (key == name) — no regression."""
        from hermes_cli.plugins_cmd import cmd_enable
        mock_user.return_value = nested_plugin_env
        mock_bundled.return_value = nested_plugin_env / "nonexistent"

        cmd_enable("disk-cleanup")
        saved = mock_save_en.call_args[0][0]
        assert "disk-cleanup" in saved

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_disabled_set", return_value={"nemo_relay"})
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_disable_bare_alias_canonicalizes_disabled_set(
        self, mock_en, mock_dis, mock_save_en, mock_save_dis,
        mock_user, mock_bundled, nested_plugin_env,
    ):
        from hermes_cli.plugins_cmd import cmd_disable
        mock_user.return_value = nested_plugin_env
        mock_bundled.return_value = nested_plugin_env / "nonexistent"

        cmd_disable("nemo_relay")

        saved_disabled = mock_save_dis.call_args[0][0]
        assert saved_disabled == {"observability/nemo_relay"}
        assert "nemo_relay" not in saved_disabled

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_disabled_set", return_value={"nemo_relay"})
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_dashboard_disable_bare_alias_canonicalizes_disabled_set(
        self, mock_en, mock_dis, mock_save_en, mock_save_dis,
        mock_user, mock_bundled, nested_plugin_env,
    ):
        from hermes_cli.plugins_cmd import dashboard_set_agent_plugin_enabled
        mock_user.return_value = nested_plugin_env
        mock_bundled.return_value = nested_plugin_env / "nonexistent"

        result = dashboard_set_agent_plugin_enabled("nemo_relay", enabled=False)

        assert result == {
            "ok": True,
            "name": "observability/nemo_relay",
            "unchanged": False,
        }
        saved_disabled = mock_save_dis.call_args[0][0]
        assert saved_disabled == {"observability/nemo_relay"}
        assert "nemo_relay" not in saved_disabled

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set())
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_dashboard_disable_bundled_platform_refuses_ignored_disabled_row(
        self, mock_en, mock_dis, mock_save_en, mock_save_dis,
        mock_user, mock_bundled, tmp_path,
    ):
        from hermes_cli.plugins_cmd import dashboard_set_agent_plugin_enabled

        bundled_dir = tmp_path / "bundled"
        _make_category_plugin(
            bundled_dir,
            "platforms",
            "discord",
            {
                "name": "discord-platform",
                "kind": "platform",
                "version": "1.0.0",
            },
        )
        user_dir = tmp_path / "user"
        user_dir.mkdir()
        mock_user.return_value = user_dir
        mock_bundled.return_value = bundled_dir

        result = dashboard_set_agent_plugin_enabled("discord-platform", enabled=False)

        assert result["ok"] is False
        assert "gateway.platforms.discord.enabled" in result["error"]
        mock_save_en.assert_not_called()
        mock_save_dis.assert_not_called()

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_disabled_set", return_value={"platforms/discord"})
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_dashboard_enable_bundled_platform_cleans_legacy_disabled_alias(
        self, mock_en, mock_dis, mock_save_en, mock_save_dis,
        mock_user, mock_bundled, tmp_path,
    ):
        from hermes_cli.plugins_cmd import dashboard_set_agent_plugin_enabled

        bundled_dir = tmp_path / "bundled"
        _make_category_plugin(
            bundled_dir,
            "platforms",
            "discord",
            {
                "name": "discord-platform",
                "kind": "platform",
                "version": "1.0.0",
            },
        )
        user_dir = tmp_path / "user"
        user_dir.mkdir()
        mock_user.return_value = user_dir
        mock_bundled.return_value = bundled_dir

        result = dashboard_set_agent_plugin_enabled("discord", enabled=True)

        assert result == {"ok": True, "name": "discord-platform", "unchanged": False}
        saved_disabled = mock_save_dis.call_args[0][0]
        assert "platforms/discord" not in saved_disabled
        mock_save_en.assert_not_called()

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set())
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_dashboard_enable_bundled_platform_returns_config_hint(
        self, mock_en, mock_dis, mock_save_en, mock_save_dis,
        mock_user, mock_bundled, tmp_path,
    ):
        from hermes_cli.plugins_cmd import dashboard_set_agent_plugin_enabled

        bundled_dir = tmp_path / "bundled"
        _make_category_plugin(
            bundled_dir,
            "platforms",
            "discord",
            {
                "name": "discord-platform",
                "kind": "platform",
                "version": "1.0.0",
            },
        )
        user_dir = tmp_path / "user"
        user_dir.mkdir()
        mock_user.return_value = user_dir
        mock_bundled.return_value = bundled_dir

        result = dashboard_set_agent_plugin_enabled("discord", enabled=True)

        assert result["ok"] is False
        assert result["reason"] == "channel_config_required"
        assert "platforms.discord.enabled" in result["hint"]
        mock_save_en.assert_not_called()
        mock_save_dis.assert_not_called()

    def test_enable_bundled_platform_preserves_disabled_user_plugin_collision(
        self, monkeypatch, tmp_path, capsys,
    ):
        import hermes_cli.plugins_cmd as pc

        bundled_dir = tmp_path / "bundled"
        _make_category_plugin(
            bundled_dir,
            "platforms",
            "discord",
            {
                "name": "discord-platform",
                "kind": "platform",
                "version": "1.0.0",
            },
        )
        user_dir = tmp_path / "user"
        _make_plugin_dir(
            user_dir,
            "discord",
            {
                "name": "discord",
                "kind": "standalone",
                "version": "1.0.0",
            },
        )
        saved = {}
        monkeypatch.setattr(
            "hermes_cli.plugins.get_bundled_plugins_dir",
            lambda: bundled_dir,
        )
        monkeypatch.setattr(pc, "_plugins_dir", lambda: user_dir)
        monkeypatch.setattr(pc, "_get_enabled_set", lambda: set())
        monkeypatch.setattr(pc, "_get_disabled_set", lambda: {"discord"})
        monkeypatch.setattr(
            pc,
            "_save_disabled_set",
            lambda value: saved.setdefault("disabled", value),
        )
        monkeypatch.setattr(
            pc,
            "_save_enabled_set",
            lambda value: saved.setdefault("enabled", value),
        )

        pc.cmd_enable("discord-platform")

        assert "disabled" not in saved
        assert "registered automatically" in capsys.readouterr().out

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_disabled_set", return_value={"discord"})
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_dashboard_enable_bundled_platform_preserves_disabled_user_plugin_collision(
        self, mock_en, mock_dis, mock_save_en, mock_save_dis,
        mock_user, mock_bundled, tmp_path,
    ):
        from hermes_cli.plugins_cmd import dashboard_set_agent_plugin_enabled

        bundled_dir = tmp_path / "bundled"
        _make_category_plugin(
            bundled_dir,
            "platforms",
            "discord",
            {
                "name": "discord-platform",
                "kind": "platform",
                "version": "1.0.0",
            },
        )
        user_dir = tmp_path / "user"
        _make_plugin_dir(
            user_dir,
            "discord",
            {
                "name": "discord",
                "kind": "standalone",
                "version": "1.0.0",
            },
        )
        mock_user.return_value = user_dir
        mock_bundled.return_value = bundled_dir

        result = dashboard_set_agent_plugin_enabled("discord-platform", enabled=True)

        assert result["ok"] is False
        assert result["reason"] == "channel_config_required"
        mock_save_en.assert_not_called()
        mock_save_dis.assert_not_called()

    def test_project_plugin_collision_is_claimed_only_when_project_plugins_enabled(
        self, monkeypatch, tmp_path,
    ):
        import hermes_cli.plugins_cmd as pc

        bundled_dir = tmp_path / "bundled"
        _make_category_plugin(
            bundled_dir,
            "platforms",
            "discord",
            {
                "name": "discord-platform",
                "kind": "platform",
                "version": "1.0.0",
            },
        )
        project_dir = tmp_path / ".hermes" / "plugins"
        _make_plugin_dir(
            project_dir,
            "discord",
            {
                "name": "discord",
                "kind": "standalone",
                "version": "1.0.0",
            },
        )
        user_dir = tmp_path / "user"
        user_dir.mkdir()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "hermes_cli.plugins.get_bundled_plugins_dir",
            lambda: bundled_dir,
        )
        monkeypatch.setattr(pc, "_plugins_dir", lambda: user_dir)
        monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)

        assert "discord" not in pc._non_bundled_plugin_config_aliases()

        monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "1")

        assert "discord" in pc._non_bundled_plugin_config_aliases()

    def test_entry_point_plugin_collision_is_claimed(self, monkeypatch, tmp_path):
        import hermes_cli.plugins_cmd as pc

        bundled_dir = tmp_path / "bundled"
        _make_category_plugin(
            bundled_dir,
            "platforms",
            "discord",
            {
                "name": "discord-platform",
                "kind": "platform",
                "version": "1.0.0",
            },
        )
        user_dir = tmp_path / "user"
        user_dir.mkdir()
        monkeypatch.setattr(
            "hermes_cli.plugins.get_bundled_plugins_dir",
            lambda: bundled_dir,
        )
        monkeypatch.setattr(pc, "_plugins_dir", lambda: user_dir)
        monkeypatch.setattr(
            pc.importlib.metadata,
            "entry_points",
            lambda: [SimpleNamespace(name="discord", group="hermes_agent.plugins")],
        )

        assert "discord" in pc._non_bundled_plugin_config_aliases()

    def test_toggle_ui_excludes_bundled_platforms(self, monkeypatch, tmp_path):
        import hermes_cli.plugins_cmd as pc

        platform_dir = tmp_path / "bundled" / "platforms" / "discord"
        _make_plugin_dir(
            platform_dir.parent,
            "discord",
            {
                "name": "discord-platform",
                "kind": "platform",
                "version": "1.0.0",
            },
        )
        normal_dir = tmp_path / "bundled" / "disk-cleanup"
        _make_plugin_dir(
            normal_dir.parent,
            "disk-cleanup",
            {
                "name": "disk-cleanup",
                "kind": "standalone",
                "version": "1.0.0",
            },
        )
        entries = [
            ("discord-platform", "1.0.0", "", "bundled", platform_dir, "discord-platform"),
            ("disk-cleanup", "1.0.0", "", "bundled", normal_dir, "disk-cleanup"),
        ]
        captured = {}

        def fake_ui(_curses, plugin_names, _labels, _selected, _disabled,
                    _categories, _console, non_toggleable_aliases):
            captured["plugin_names"] = plugin_names
            captured["non_toggleable_aliases"] = non_toggleable_aliases

        monkeypatch.setattr(pc, "_discover_all_plugins", lambda: entries)
        monkeypatch.setattr(pc, "_get_enabled_set", lambda: set())
        monkeypatch.setattr(pc, "_get_disabled_set", lambda: {"platforms/discord"})
        monkeypatch.setattr(pc, "_get_current_memory_provider", lambda: "")
        monkeypatch.setattr(pc, "_get_current_context_engine", lambda: "compressor")
        monkeypatch.setattr(pc, "_run_composite_ui", fake_ui)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.setitem(sys.modules, "curses", SimpleNamespace())

        pc.cmd_toggle()

        assert captured["plugin_names"] == ["disk-cleanup"]
        assert "platforms/discord" in captured["non_toggleable_aliases"]

    def test_toggle_ui_uses_canonical_keys_for_nested_plugins(self, monkeypatch, tmp_path):
        import hermes_cli.plugins_cmd as pc

        nested_dir = tmp_path / "user" / "observability" / "nemo_relay"
        _make_plugin_dir(
            nested_dir.parent,
            "nemo_relay",
            {
                "name": "nemo_relay",
                "kind": "standalone",
                "version": "1.0.0",
            },
        )
        entries = [
            (
                "nemo_relay",
                "1.0.0",
                "",
                "user",
                nested_dir,
                "observability/nemo_relay",
            )
        ]
        captured = {}

        def fake_ui(_curses, plugin_names, _labels, _selected, _disabled,
                    _categories, _console, _non_toggleable_aliases):
            captured["plugin_names"] = plugin_names

        monkeypatch.setattr(pc, "_discover_all_plugins", lambda: entries)
        monkeypatch.setattr(pc, "_get_enabled_set", lambda: set())
        monkeypatch.setattr(pc, "_get_disabled_set", lambda: set())
        monkeypatch.setattr(pc, "_get_current_memory_provider", lambda: "")
        monkeypatch.setattr(pc, "_get_current_context_engine", lambda: "compressor")
        monkeypatch.setattr(pc, "_run_composite_ui", fake_ui)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.setitem(sys.modules, "curses", SimpleNamespace())

        pc.cmd_toggle()

        assert captured["plugin_names"] == ["observability/nemo_relay"]

    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_fallback_toggle_strips_bundled_platform_aliases(
        self, mock_get_en, mock_save_en, mock_save_dis, monkeypatch,
    ):
        import hermes_cli.plugins_cmd as pc

        monkeypatch.setattr("builtins.input", lambda _prompt="": "")

        pc._run_composite_fallback(
            ["disk-cleanup"],
            ["disk-cleanup"],
            set(),
            {"platforms/discord"},
            [],
            SimpleNamespace(print=lambda *args, **kwargs: None),
            {"platforms/discord"},
        )

        saved_disabled = mock_save_dis.call_args[0][0]
        assert "platforms/discord" not in saved_disabled
        assert saved_disabled == {"disk-cleanup"}

    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value={"disk-cleanup"})
    def test_fallback_toggle_strips_bundled_platform_aliases_without_general_plugins(
        self, mock_get_en, mock_save_en, mock_save_dis, monkeypatch,
    ):
        import hermes_cli.plugins_cmd as pc

        monkeypatch.setattr("builtins.input", lambda _prompt="": "")

        pc._run_composite_fallback(
            [],
            [],
            set(),
            {"platforms/discord"},
            [],
            SimpleNamespace(print=lambda *args, **kwargs: None),
            {"platforms/discord"},
        )

        assert mock_save_en.call_args[0][0] == {"disk-cleanup"}
        assert mock_save_dis.call_args[0][0] == set()
