"""API endpoints for system and per-user AI model selection."""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..auth import get_current_user, require_instructor
from ..services import model_settings

router = APIRouter()


class ModelDefaultsUpdate(BaseModel):
  provider: str = "anthropic"
  models: dict[str, Optional[str]] = Field(..., min_length=1)


@router.get("")
async def get_settings(current_user: dict = Depends(get_current_user)):
  return {
    "providers": list(model_settings.ai_helper.MODEL_CONFIG),
    "settings": model_settings.get_effective_settings(current_user["user_id"]),
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
