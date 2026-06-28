"""Tests for the nested category plugin discovery fix (issue #41066).

Verifies that _discover_all_plugins() recurses into category directories
(up to 2 levels deep) and that _plugin_status() checks both manifest name
and path-derived key against the enabled/disabled sets.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plugin_dir(parent: Path, name: str, manifest: dict) -> Path:
    """Create a minimal plugin directory with a plugin.yaml."""
    d = parent / name
    d.mkdir(parents=True, exist_ok=True)
    import yaml
    (d / "plugin.yaml").write_text(yaml.dump(manifest), encoding="utf-8")
    (d / "__init__.py").write_text("def register(ctx): pass\n", encoding="utf-8")
    return d


def _make_category_plugin(
    parent: Path, category: str, name: str, manifest: dict
) -> Path:
    """Create a category-namespaced plugin: <parent>/<category>/<name>/plugin.yaml."""
    return _make_plugin_dir(parent / category, name, manifest)


# ---------------------------------------------------------------------------
# _read_manifest_info
# ---------------------------------------------------------------------------


class TestReadManifestInfo:
    def test_flat_plugin(self, tmp_path):
        from hermes_cli.plugins_cmd import _read_manifest_info

        d = _make_plugin_dir(tmp_path, "my-plugin", {
            "name": "my-plugin", "version": "1.0.0", "description": "test"
        })
        result = _read_manifest_info(d, "")
        assert result is not None
        name, version, description, key = result
        assert name == "my-plugin"
        assert version == "1.0.0"
        assert description == "test"
        assert key == "my-plugin"  # flat: key == name

    def test_category_plugin(self, tmp_path):
        from hermes_cli.plugins_cmd import _read_manifest_info

        d = _make_category_plugin(tmp_path, "web", "tavily", {
            "name": "web-tavily", "version": "2.0.0", "description": "search"
        })
        result = _read_manifest_info(d, "web")
        assert result is not None
        name, version, description, key = result
        assert name == "web-tavily"  # manifest name
        assert key == "web/tavily"  # path-derived key

    def test_no_manifest(self, tmp_path):
        from hermes_cli.plugins_cmd import _read_manifest_info

        d = tmp_path / "empty-dir"
        d.mkdir()
        assert _read_manifest_info(d, "") is None

    def test_yml_extension(self, tmp_path):
        from hermes_cli.plugins_cmd import _read_manifest_info

        d = tmp_path / "my-plugin"
        d.mkdir()
        import yaml
        (d / "plugin.yml").write_text(yaml.dump({"name": "my-plugin"}), encoding="utf-8")
        result = _read_manifest_info(d, "")
        assert result is not None
        assert result[0] == "my-plugin"


# ---------------------------------------------------------------------------
# _discover_all_plugins — recursive discovery
# ---------------------------------------------------------------------------


class TestDiscoverAllPlugins:
    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_flat_plugins_still_discovered(self, mock_user_dir, mock_bundled_dir, tmp_path):
        from hermes_cli.plugins_cmd import _discover_all_plugins

        _make_plugin_dir(tmp_path, "disk-cleanup", {
            "name": "disk-cleanup", "version": "1.0.0"
        })
        mock_user_dir.return_value = tmp_path
        mock_bundled_dir.return_value = tmp_path / "nonexistent"

        entries = _discover_all_plugins()
        keys = [e[5] for e in entries]
        assert "disk-cleanup" in keys

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_category_plugins_discovered(self, mock_user_dir, mock_bundled_dir, tmp_path):
        from hermes_cli.plugins_cmd import _discover_all_plugins

        _make_category_plugin(tmp_path, "web", "tavily", {
            "name": "web-tavily", "version": "1.0.0"
        })
        _make_category_plugin(tmp_path, "image_gen", "openai", {
            "name": "image-gen-openai", "version": "2.0.0"
        })
        mock_user_dir.return_value = tmp_path
        mock_bundled_dir.return_value = tmp_path / "nonexistent"

        entries = _discover_all_plugins()
        keys = [e[5] for e in entries]
        assert "web/tavily" in keys
        assert "image_gen/openai" in keys

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_mixed_flat_and_category(self, mock_user_dir, mock_bundled_dir, tmp_path):
        from hermes_cli.plugins_cmd import _discover_all_plugins

        _make_plugin_dir(tmp_path, "disk-cleanup", {
            "name": "disk-cleanup", "version": "1.0.0"
        })
        _make_category_plugin(tmp_path, "web", "tavily", {
            "name": "web-tavily", "version": "1.0.0"
        })
        _make_category_plugin(tmp_path, "web", "exa", {
            "name": "web-exa", "version": "1.0.0"
        })
        mock_user_dir.return_value = tmp_path
        mock_bundled_dir.return_value = tmp_path / "nonexistent"

        entries = _discover_all_plugins()
        keys = [e[5] for e in entries]
        assert "disk-cleanup" in keys
        assert "web/tavily" in keys
        assert "web/exa" in keys
        assert len(entries) == 3

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_depth_cap_at_two(self, mock_user_dir, mock_bundled_dir, tmp_path):
        """Plugins nested 3 levels deep should NOT be discovered."""
        from hermes_cli.plugins_cmd import _discover_all_plugins

        # 2 levels: should be found
        _make_category_plugin(tmp_path, "web", "tavily", {
            "name": "web-tavily", "version": "1.0.0"
        })
        # 3 levels: should NOT be found
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        import yaml
        (deep / "plugin.yaml").write_text(
            yaml.dump({"name": "too-deep"}), encoding="utf-8"
        )
        mock_user_dir.return_value = tmp_path
        mock_bundled_dir.return_value = tmp_path / "nonexistent"

        entries = _discover_all_plugins()
        keys = [e[5] for e in entries]
        assert "web/tavily" in keys
        assert "a/b/c" not in keys

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_tuple_has_six_elements(self, mock_user_dir, mock_bundled_dir, tmp_path):
        from hermes_cli.plugins_cmd import _discover_all_plugins

        _make_category_plugin(tmp_path, "web", "tavily", {
            "name": "web-tavily", "version": "1.0.0", "description": "search"
        })
        mock_user_dir.return_value = tmp_path
        mock_bundled_dir.return_value = tmp_path / "nonexistent"

        entries = _discover_all_plugins()
        assert len(entries) == 1
        entry = entries[0]
        assert len(entry) == 6
        name, version, description, source, dir_path, key = entry
        assert name == "web-tavily"
        assert key == "web/tavily"
        assert source == "user"

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_user_overrides_bundled_on_key_collision(self, mock_user_dir, mock_bundled_dir, tmp_path):
        """User plugin with same key as bundled should win."""
        from hermes_cli.plugins_cmd import _discover_all_plugins

        # Simulate a bundled plugin
        bundled_dir = tmp_path / "bundled"
        bundled_dir.mkdir()
        _make_plugin_dir(bundled_dir, "my-plugin", {
            "name": "my-plugin", "version": "1.0.0"
        })
        # User plugin with same key
        _make_plugin_dir(tmp_path, "my-plugin", {
            "name": "my-plugin", "version": "2.0.0"
        })
        mock_user_dir.return_value = tmp_path
        mock_bundled_dir.return_value = bundled_dir

        entries = _discover_all_plugins()
        keys = [e[5] for e in entries]
        assert keys.count("my-plugin") == 1
        # User version should win
        entry = [e for e in entries if e[5] == "my-plugin"][0]
        assert entry[1] == "2.0.0"

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_bundled_platform_key_matches_runtime_loader(
        self, mock_user_dir, mock_bundled_dir, tmp_path
    ):
        """Bundled platforms use manifest names, not platforms/<dir> keys."""
        from hermes_cli.plugins_cmd import _discover_all_plugins

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
        mock_user_dir.return_value = user_dir
        mock_bundled_dir.return_value = bundled_dir

        entries = _discover_all_plugins()
        keys = [e[5] for e in entries]
        assert "discord-platform" in keys
        assert "platforms/discord" not in keys


# ---------------------------------------------------------------------------
# _plugin_status — key-aware status
# ---------------------------------------------------------------------------


class TestPluginStatus:
    def test_name_in_enabled(self):
        from hermes_cli.plugins_cmd import _plugin_status
        assert _plugin_status("my-plugin", {"my-plugin"}, set()) == "enabled"

    def test_key_in_enabled(self):
        from hermes_cli.plugins_cmd import _plugin_status
        assert _plugin_status("web-tavily", {"web/tavily"}, set(), key="web/tavily") == "enabled"

    def test_name_in_disabled(self):
        from hermes_cli.plugins_cmd import _plugin_status
        assert _plugin_status("my-plugin", set(), {"my-plugin"}) == "disabled"

    def test_key_in_disabled(self):
        from hermes_cli.plugins_cmd import _plugin_status
        assert _plugin_status("web-tavily", set(), {"web/tavily"}, key="web/tavily") == "disabled"

    def test_bundled_platform_config_keys_use_declared_gateway_keys(self, tmp_path):
        from hermes_cli.plugins_cmd import _bundled_platform_config_keys

        platform_dir = _make_plugin_dir(tmp_path, "wecom", {
            "name": "wecom-platform",
            "kind": "platform",
            "version": "1.0.0",
            "gateway_keys": ["wecom", "wecom_callback"],
        })
        entry = (
            "wecom-platform",
            "1.0.0",
            "",
            "bundled",
            platform_dir,
            "wecom-platform",
        )

        assert _bundled_platform_config_keys(entry) == ("wecom", "wecom_callback")

    def test_multi_key_platform_status_requires_all_keys_disabled(self, tmp_path):
        from hermes_cli.plugins_cmd import _entry_status

        platform_dir = _make_plugin_dir(tmp_path, "wecom", {
            "name": "wecom-platform",
            "kind": "platform",
            "version": "1.0.0",
            "gateway_keys": ["wecom", "wecom_callback"],
        })
        entry = (
            "wecom-platform",
            "1.0.0",
            "",
            "bundled",
            platform_dir,
            "wecom-platform",
        )

        assert _entry_status(
            entry,
            set(),
            set(),
            {"platforms": {"wecom": {"enabled": False}}},
        ) == "enabled"
        assert _entry_status(
            entry,
            set(),
            set(),
            {
                "platforms": {
                    "wecom": {"enabled": False},
                    "wecom_callback": {"enabled": False},
                },
            },
        ) == "disabled"

    def test_neither_name_nor_key(self):
        from hermes_cli.plugins_cmd import _plugin_status
        assert _plugin_status("unknown", {"other"}, set(), key="cat/unknown") == "not enabled"

    def test_disabled_takes_precedence_over_enabled(self):
        from hermes_cli.plugins_cmd import _plugin_status
        assert _plugin_status("my-plugin", {"my-plugin"}, {"my-plugin"}) == "disabled"

    def test_key_disabled_takes_precedence(self):
        from hermes_cli.plugins_cmd import _plugin_status
        assert _plugin_status("web-tavily", {"web/tavily"}, {"web/tavily"}, key="web/tavily") == "disabled"

    def test_bundled_platform_entry_ignores_stale_disabled(self, tmp_path):
        from hermes_cli.plugins_cmd import _entry_status

        platform_dir = tmp_path / "platforms" / "discord"
        _make_plugin_dir(platform_dir.parent, "discord", {
            "name": "discord-platform",
            "kind": "platform",
        })
        entry = (
            "discord-platform",
            "1.0.0",
            "Discord",
            "bundled",
            platform_dir,
            "discord-platform",
        )

        assert _entry_status(entry, set(), {"discord-platform"}) == "enabled"

    def test_bundled_platform_aliases_include_channel_and_legacy_key(self, tmp_path):
        from hermes_cli.plugins_cmd import _plugin_config_aliases_for_entry

        platform_dir = tmp_path / "platforms" / "discord"
        _make_plugin_dir(platform_dir.parent, "discord", {
            "name": "discord-platform",
            "kind": "platform",
        })
        entry = (
            "discord-platform",
            "1.0.0",
            "Discord",
            "bundled",
            platform_dir,
            "discord-platform",
        )

        aliases = _plugin_config_aliases_for_entry(entry)
        assert "discord" in aliases
        assert "platforms/discord" in aliases

    def test_bundled_platform_entry_accepts_yml_manifest(self, tmp_path):
        from hermes_cli.plugins_cmd import _is_bundled_platform_entry, _read_manifest

        platform_dir = tmp_path / "platforms" / "discord"
        platform_dir.mkdir(parents=True)
        (platform_dir / "plugin.yml").write_text(
            "name: discord-platform\nkind: platform\n",
            encoding="utf-8",
        )
        _read_manifest.cache_clear()
        entry = (
            "discord-platform",
            "1.0.0",
            "Discord",
            "bundled",
            platform_dir,
            "discord-platform",
        )

        assert _is_bundled_platform_entry(entry)


# ---------------------------------------------------------------------------
# Integration: _filter_plugin_entries with category plugins
# ---------------------------------------------------------------------------


class TestFilterPluginEntries:
    def test_enabled_filter_uses_key(self):
        from hermes_cli.plugins_cmd import _filter_plugin_entries

        entries = [
            ("web-tavily", "1.0.0", "search", "user", Path("/tmp"), "web/tavily"),
            ("disk-cleanup", "1.0.0", "cleanup", "bundled", Path("/tmp"), "disk-cleanup"),
        ]
        args = MagicMock()
        args.no_bundled = False
        args.user = False
        args.enabled = True

        result = _filter_plugin_entries(entries, args, {"web/tavily"}, set())
        assert len(result) == 1
        assert result[0][5] == "web/tavily"

    def test_enabled_filter_by_name_still_works(self):
        from hermes_cli.plugins_cmd import _filter_plugin_entries

        entries = [
            ("disk-cleanup", "1.0.0", "cleanup", "bundled", Path("/tmp"), "disk-cleanup"),
        ]
        args = MagicMock()
        args.no_bundled = False
        args.user = False
        args.enabled = True

        result = _filter_plugin_entries(entries, args, {"disk-cleanup"}, set())
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Integration: cmd_list JSON output includes category plugins
# ---------------------------------------------------------------------------


class TestCmdListJson:
    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_json_output_includes_category_plugins(self, mock_user_dir, mock_bundled_dir, tmp_path, capsys):
        from hermes_cli.plugins_cmd import cmd_list

        _make_category_plugin(tmp_path, "web", "tavily", {
            "name": "web-tavily", "version": "1.0.0", "description": "search"
        })
        _make_plugin_dir(tmp_path, "disk-cleanup", {
            "name": "disk-cleanup", "version": "2.0.0", "description": "cleanup"
        })
        mock_user_dir.return_value = tmp_path
        mock_bundled_dir.return_value = tmp_path / "nonexistent"

        args = MagicMock()
        args.json = True
        args.plain = False
        args.no_bundled = False
        args.user = False
        args.enabled = False

        cmd_list(args)
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        names = [p["name"] for p in payload]
        assert "web-tavily" in names
        assert "disk-cleanup" in names

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_json_status_uses_key(self, mock_user_dir, mock_bundled_dir, tmp_path, capsys):
        from hermes_cli.plugins_cmd import cmd_list

        _make_category_plugin(tmp_path, "web", "tavily", {
            "name": "web-tavily", "version": "1.0.0"
        })
        mock_user_dir.return_value = tmp_path
        mock_bundled_dir.return_value = tmp_path / "nonexistent"

        # Patch config to return web/tavily as enabled
        with patch("hermes_cli.plugins_cmd._get_enabled_set", return_value={"web/tavily"}):
            args = MagicMock()
            args.json = True
            args.plain = False
            args.no_bundled = False
            args.user = False
            args.enabled = False

            cmd_list(args)
            captured = capsys.readouterr()
            payload = json.loads(captured.out)
            assert len(payload) == 1
            assert payload[0]["status"] == "enabled"

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_json_status_ignores_stale_disabled_for_bundled_platform(
        self, mock_user_dir, mock_bundled_dir, tmp_path, capsys
    ):
        from hermes_cli.plugins_cmd import cmd_list

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
        mock_user_dir.return_value = user_dir
        mock_bundled_dir.return_value = bundled_dir

        with patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set()), \
             patch("hermes_cli.plugins_cmd._get_disabled_set", return_value={"discord-platform"}):
            args = MagicMock()
            args.json = True
            args.plain = False
            args.no_bundled = False
            args.user = False
            args.enabled = False

            cmd_list(args)
            captured = capsys.readouterr()
            payload = json.loads(captured.out)
            assert payload == [
                {
                    "name": "discord-platform",
                    "status": "enabled",
                    "version": "1.0.0",
                    "description": "",
                    "source": "bundled",
                }
            ]

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_json_status_respects_gateway_disabled_for_bundled_platform(
        self, mock_user_dir, mock_bundled_dir, tmp_path, capsys
    ):
        from hermes_cli.plugins_cmd import cmd_list

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
        mock_user_dir.return_value = user_dir
        mock_bundled_dir.return_value = bundled_dir

        with patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set()), \
             patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set()), \
             patch("hermes_cli.plugins_cmd._load_status_config", return_value={
                 "gateway": {"platforms": {"discord": {"enabled": False}}},
             }):
            args = MagicMock()
            args.json = True
            args.plain = False
            args.no_bundled = False
            args.user = False
            args.enabled = False

            cmd_list(args)
            captured = capsys.readouterr()
            payload = json.loads(captured.out)
            assert payload[0]["name"] == "discord-platform"
            assert payload[0]["status"] == "disabled"

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_json_status_respects_top_level_platform_disabled_for_bundled_platform(
        self, mock_user_dir, mock_bundled_dir, tmp_path, capsys
    ):
        from hermes_cli.plugins_cmd import cmd_list

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
        mock_user_dir.return_value = user_dir
        mock_bundled_dir.return_value = bundled_dir

        with patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set()), \
             patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set()), \
             patch("hermes_cli.plugins_cmd._load_status_config", return_value={
                 "platforms": {"discord": {"enabled": False}},
             }):
            args = MagicMock()
            args.json = True
            args.plain = False
            args.no_bundled = False
            args.user = False
            args.enabled = False

            cmd_list(args)
            captured = capsys.readouterr()
            payload = json.loads(captured.out)
            assert payload[0]["name"] == "discord-platform"
            assert payload[0]["status"] == "disabled"

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_json_status_top_level_platform_config_overrides_gateway_platform(
        self, mock_user_dir, mock_bundled_dir, tmp_path, capsys
    ):
        from hermes_cli.plugins_cmd import cmd_list

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
        mock_user_dir.return_value = user_dir
        mock_bundled_dir.return_value = bundled_dir

        with patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set()), \
             patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set()), \
             patch("hermes_cli.plugins_cmd._load_status_config", return_value={
                 "gateway": {"platforms": {"discord": {"enabled": False}}},
                 "platforms": {"discord": {"enabled": True}},
             }):
            args = MagicMock()
            args.json = True
            args.plain = False
            args.no_bundled = False
            args.user = False
            args.enabled = False

            cmd_list(args)
            captured = capsys.readouterr()
            payload = json.loads(captured.out)
            assert payload[0]["name"] == "discord-platform"
            assert payload[0]["status"] == "enabled"

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @pytest.mark.parametrize("enabled_value", ["false", "0", "no", "off", "maybe"])
    def test_json_status_respects_direct_top_level_platform_bool_string(
        self, mock_user_dir, mock_bundled_dir, tmp_path, capsys, enabled_value
    ):
        from hermes_cli.plugins_cmd import cmd_list

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
        mock_user_dir.return_value = user_dir
        mock_bundled_dir.return_value = bundled_dir

        with patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set()), \
             patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set()), \
             patch("hermes_cli.plugins_cmd._load_status_config", return_value={
                 "discord": {"enabled": enabled_value},
                 "platforms": {"discord": {"enabled": True}},
             }):
            args = MagicMock()
            args.json = True
            args.plain = False
            args.no_bundled = False
            args.user = False
            args.enabled = False

            cmd_list(args)
            captured = capsys.readouterr()
            payload = json.loads(captured.out)
            assert payload[0]["name"] == "discord-platform"
            assert payload[0]["status"] == "disabled"

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_json_status_uses_manifest_gateway_key_for_bundled_platform(
        self, mock_user_dir, mock_bundled_dir, tmp_path, capsys
    ):
        from hermes_cli.plugins_cmd import cmd_list

        bundled_dir = tmp_path / "bundled"
        _make_category_plugin(
            bundled_dir,
            "platforms",
            "directory-name",
            {
                "name": "custom-platform",
                "gateway_key": "custom_gateway",
                "kind": "platform",
                "version": "1.0.0",
            },
        )
        user_dir = tmp_path / "user"
        user_dir.mkdir()
        mock_user_dir.return_value = user_dir
        mock_bundled_dir.return_value = bundled_dir

        with patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set()), \
             patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set()), \
             patch("hermes_cli.plugins_cmd._load_status_config", return_value={
                 "platforms": {"custom_gateway": {"enabled": False}},
             }):
            args = MagicMock()
            args.json = True
            args.plain = False
            args.no_bundled = False
            args.user = False
            args.enabled = False

            cmd_list(args)
            captured = capsys.readouterr()
            payload = json.loads(captured.out)
            assert payload[0]["name"] == "custom-platform"
            assert payload[0]["status"] == "disabled"
