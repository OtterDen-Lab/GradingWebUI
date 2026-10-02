"""API endpoints for system and per-user AI model selection."""
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from ..auth import get_current_user, require_instructor
from ..services import model_settings
from ..services import ollama_settings

router = APIRouter()


class ModelDefaultsUpdate(BaseModel):
  provider: str = "anthropic"
  models: dict[str, Optional[str]] = Field(..., min_length=1)


class OllamaServerUpdate(BaseModel):
  name: str = Field(..., min_length=1, max_length=120)
  base_url: str = Field(..., min_length=8, max_length=500)


class OllamaModelUpdate(BaseModel):
  model: Optional[str] = Field(None, max_length=240)


@router.get("")
async def get_settings(current_user: dict = Depends(get_current_user)):
  ollama_server = ollama_settings.get_active_server()
  return {
    "providers": list(model_settings.ai_helper.MODEL_CONFIG),
    "settings": model_settings.get_effective_settings(current_user["user_id"]),
    "ollama_active": ({"server_name": ollama_server["name"],
                       "model_id": ollama_server["active_model"]}
                      if ollama_server else None),
  }


@router.get("/models/{provider}")
async def get_models(provider: str, current_user: dict = Depends(require_instructor)):
  try:
    return {"provider": provider, "models": model_settings.list_provider_models(provider)}
  except ValueError as error:
    raise HTTPException(status_code=400, detail=str(error)) from error
  except Exception as error:
    raise HTTPException(status_code=502, detail=f"Could not retrieve provider models: {error}") from error


@router.put("/system")
async def update_system_settings(request: ModelDefaultsUpdate,
                                 current_user: dict = Depends(require_instructor)):
  try:
    model_settings.set_system_defaults(request.provider, request.models,
                                       current_user["user_id"])
  except ValueError as error:
    raise HTTPException(status_code=400, detail=str(error)) from error
  return {"settings": model_settings.get_effective_settings(current_user["user_id"], request.provider)}


@router.put("/me")
async def update_my_settings(request: ModelDefaultsUpdate,
                             current_user: dict = Depends(get_current_user)):
  try:
    model_settings.set_user_overrides(current_user["user_id"], request.provider,
                                      request.models)
  except ValueError as error:
    raise HTTPException(status_code=400, detail=str(error)) from error
  return {"settings": model_settings.get_effective_settings(current_user["user_id"], request.provider)}


@router.get("/ollama/servers")
async def get_ollama_servers(current_user: dict = Depends(require_instructor)):
  return {"servers": ollama_settings.list_servers()}


@router.post("/ollama/servers")
async def create_ollama_server(request: OllamaServerUpdate,
                               current_user: dict = Depends(require_instructor)):
  try:
    return {"server": ollama_settings.save_server(request.name, request.base_url,
                                                     current_user["user_id"])}
  except ValueError as error:
    raise HTTPException(status_code=400, detail=str(error)) from error


@router.put("/ollama/servers/{server_id}")
async def update_ollama_server(server_id: int, request: OllamaServerUpdate,
                               current_user: dict = Depends(require_instructor)):
  try:
    return {"server": ollama_settings.save_server(request.name, request.base_url,
                                                     current_user["user_id"], server_id)}
  except ValueError as error:
    raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/ollama/servers/{server_id}/models")
async def get_ollama_models(server_id: int,
                            current_user: dict = Depends(require_instructor)):
  try:
    return {"models": ollama_settings.list_models(server_id)}
  except ValueError as error:
    raise HTTPException(status_code=404, detail=str(error)) from error
  except RuntimeError as error:
    raise HTTPException(status_code=502, detail=str(error)) from error


@router.put("/ollama/servers/{server_id}/active-model")
async def update_ollama_active_model(server_id: int, request: OllamaModelUpdate,
                                     current_user: dict = Depends(require_instructor)):
  try:
    return {"server": ollama_settings.set_active_model(server_id, request.model,
                                                          current_user["user_id"])}
  except ValueError as error:
    raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/ollama/servers/{server_id}/pull")
async def pull_ollama_model(server_id: int, request: OllamaModelUpdate,
                            background_tasks: BackgroundTasks,
                            current_user: dict = Depends(require_instructor)):
  try:
    if not (request.model or "").strip():
      raise ValueError("An Ollama model name/tag is required")
    if not ollama_settings.get_server(server_id):
      raise ValueError("Ollama server not found")
    # Pulling a 30B model can take a long time; never hold the settings request open.
    background_tasks.add_task(ollama_settings.pull_model, server_id, request.model or "")
  except ValueError as error:
    raise HTTPException(status_code=400, detail=str(error)) from error
  return {"status": "started", "message": "Ollama model pull started; refresh models when it completes."}
