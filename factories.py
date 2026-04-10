from typing import List
from cat import hook, BillTheLizard, EmbedderSettings, LLMSettings, log
from cat.db.cruds import settings as crud_settings

from .embedders.configs import EmbedderOllamaConfig
from .llms.configs import LLMOllamaConfig


@hook(priority=1)
def factory_allowed_llms(allowed: List[LLMSettings], cat) -> List:
    return allowed + [LLMOllamaConfig]


@hook(priority=1)
def factory_allowed_embedders(allowed: List[EmbedderSettings], lizard) -> List:
    return allowed + [EmbedderOllamaConfig]


@hook(priority=1)
async def lizard_notify_plugin_installation(plugin_id: str, plugin_path: str, lizard: BillTheLizard) -> None:
    this_plugin_id = lizard.mad_hatter.get_plugin().id
    if this_plugin_id != plugin_id:
        log.warning(
            f"Plugin id mismatch: Expected {plugin_id}, got {this_plugin_id} when installing {plugin_path}"
        )
        return

    # for each Cheshire Cat, activate this plugin
    ccat_ids = await crud_settings.get_agents_main_keys()
    for ccat_id in ccat_ids:
        if (ccat := await lizard.get_cheshire_cat(ccat_id)) is None:
            continue

        await ccat.plugin_manager.toggle_plugin(plugin_id)
