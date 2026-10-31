"""API endpoints for system and per-user AI model selection."""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..auth import get_current_user, require_instructor
from ..services import model_settings

router = APIRouter()


class DefaultModelUpdate(BaseModel):
  model_id: Optional[str] = Field(None, max_length=240)


@router.get("")
async def get_settings(current_user: dict = Depends(get_current_user)):
  return {
    "default_model": model_settings.get_default_model(current_user["user_id"]),
  }


@router.get("/models/{provider}")
async def get_models(provider: str, current_user: dict = Depends(get_current_user)):
  try:
    return {"provider": provider, "models": model_settings.list_provider_models(provider)}
  except ValueError as error:
    raise HTTPException(status_code=400, detail=str(error)) from error
  except Exception as error:
    raise HTTPException(status_code=502, detail=f"Could not retrieve provider models: {error}") from error


@router.put("/system")
async def update_system_settings(request: DefaultModelUpdate,
                                 current_user: dict = Depends(require_instructor)):
  try:
    model_settings.set_default_model(current_user["user_id"], request.model_id,
                                     system=True)
  except ValueError as error:
    raise HTTPException(status_code=400, detail=str(error)) from error
  return {"default_model": model_settings.get_default_model(current_user["user_id"])}

@router.put("/me")
async def update_my_settings(request: DefaultModelUpdate,
                             current_user: dict = Depends(get_current_user)):
  try:
    model_settings.set_default_model(current_user["user_id"], request.model_id)
  except ValueError as error:
    raise HTTPException(status_code=400, detail=str(error)) from error
  return {"default_model": model_settings.get_default_model(current_user["user_id"])}
